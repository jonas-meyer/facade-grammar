"""Per-facade SAM 3 segmentation: pick the best candidate photo and persist its mask.

Panos get reprojected to a rectilinear view facing the target facade; the
projected footprint bbox is passed to SAM as a visual prompt so it locks
onto our target house in a multi-house scene.
"""

import itertools
import logging
import math
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, NamedTuple

import httpx
import numpy as np
from hamilton.function_modifiers import tag
from hamilton.htypes import Collect, Parallelizable
from PIL import Image
from pydantic import SecretStr

from facade_grammar.clients import mapillary
from facade_grammar.clients import sam as sam_client
from facade_grammar.clients.sam import BoolMask, SamInstance, SamPrompt
from facade_grammar.config import SamConfig
from facade_grammar.geo import WGS84_GEOD
from facade_grammar.schemas.buildings import (
    FEATURE_CLASS_PROMPT,
    AuditRecord,
    AuditStatus,
    Building,
    Facade,
    FacadeFeatures,
    FacadeMask,
    FacadeWork,
    FeatureClass,
    FeatureInstance,
)
from facade_grammar.schemas.photos import PhotoMetadata
from facade_grammar.viz.panorama import rectilinear_view, world_to_pixel

log = logging.getLogger(__name__)


def _audit(
    building_id: str,
    *,
    status: AuditStatus,
    photo_id: str | None = None,
    reason: str | None = None,
) -> AuditRecord:
    return AuditRecord(
        building_id=building_id,
        status=status,
        photo_id=photo_id,
        reason=reason,
        ts=datetime.now(UTC),
    )


class _Candidate(NamedTuple):
    photo: PhotoMetadata
    rank: int
    view_bytes: bytes
    facade_mask: BoolMask
    facade_bbox: tuple[int, int, int, int]
    occluder_mask: BoolMask
    facade_score: float
    occluder_ratio: float
    projected_bbox: tuple[int, int, int, int] | None
    # One entry per ``sam.feature_prompts`` class, aligned by index.
    feature_instances_by_class: list[list[SamInstance]]


@tag(stage="vision")
def per_facade_photos(
    top_photos_per_facade: dict[str, list[PhotoMetadata]],
    canal_facades: list[Facade],
    sam: SamConfig,
) -> Parallelizable[tuple[Facade, list[PhotoMetadata]]]:
    """Fan out: one parallel task per canal-front facade, optionally capped for testing."""
    facades_by_bid = {f.building_id: f for f in canal_facades}
    items = [(facades_by_bid[bid], photos) for bid, photos in top_photos_per_facade.items()]
    if sam.max_buildings is not None:
        items = items[: sam.max_buildings]
    yield from items


@tag(stage="vision")
def chosen_work_for_facade(
    per_facade_photos: tuple[Facade, list[PhotoMetadata]],
    raw_buildings: list[Building],
    sam: SamConfig,
    mapillary_token: SecretStr,
) -> FacadeWork:
    """Run SAM 3 against the top-K candidates for one facade; persist the winner's
    facade masks AND derive sub-feature instances from the same SAM call.

    Always returns a ``FacadeWork`` — never ``None``, never raises. On any
    failure, ``mask``/``features`` are ``None`` and the audit record explains
    why. That lets the ``audit_jsonl`` writer reconstruct the full outcome log
    by collecting records across cached and freshly computed facades alike.
    """
    facade, photos = per_facade_photos
    try:
        building = _lookup_building(raw_buildings, facade.building_id)
        candidates = list(_evaluate_candidates(facade, building, photos, sam, mapillary_token))
        best = _pick_best(candidates, sam.max_occluder_ratio)
        if best is None:
            return FacadeWork(
                audit=_audit(
                    facade.building_id,
                    status="no_candidates",
                    reason=f"{len(candidates)}/{len(photos)} photos produced a facade mask",
                )
            )
        sam.mask_dir.mkdir(parents=True, exist_ok=True)
        view_path = sam.mask_dir / f"{facade.building_id}_view.jpg"
        facade_path = sam.mask_dir / f"{facade.building_id}_facade.png"
        occluder_path = sam.mask_dir / f"{facade.building_id}_occluder.png"
        view_path.write_bytes(best.view_bytes)
        _write_mask(best.facade_mask, facade_path)
        _write_mask(best.occluder_mask, occluder_path)
        mask = FacadeMask(
            building_id=facade.building_id,
            photo_id=best.photo.photo_id,
            view_path=view_path,
            mask_path=facade_path,
            occluder_mask_path=occluder_path,
            facade_score=best.facade_score,
            occluder_ratio=best.occluder_ratio,
            projected_bbox=best.projected_bbox,
            facade_bbox_px=best.facade_bbox,
        )
        features = _features_from_winner(best, facade.building_id, view_path, sam)
        return FacadeWork(
            audit=_audit(
                facade.building_id,
                status="ok",
                photo_id=best.photo.photo_id,
                reason=f"n_features={len(features.instances) if features else 0}",
            ),
            mask=mask,
            features=features,
        )
    except Exception as exc:
        log.exception("chosen_work_for_facade failed for %s", facade.building_id)
        return FacadeWork(audit=_audit(facade.building_id, status="error", reason=repr(exc)))


def _features_from_winner(
    winner: _Candidate, building_id: str, view_path: Path, cfg: SamConfig
) -> FacadeFeatures:
    """Post-filter the winner's feature detections + persist per-class mask PNGs."""
    out_dir = cfg.features_dir / building_id
    out_dir.mkdir(parents=True, exist_ok=True)
    instances: list[FeatureInstance] = []
    class_mask_paths: dict[FeatureClass, Path] = {}
    for cls, class_instances in zip(
        cfg.feature_prompts, winner.feature_instances_by_class, strict=True
    ):
        kept = _filter_instances(class_instances, winner.facade_bbox, cfg.feature_min_score)
        mask_path = out_dir / f"{cls}.png"
        _write_mask(_union_mask(kept, shape=winner.facade_mask.shape), mask_path)
        class_mask_paths[cls] = mask_path
        for inst in kept:
            x0, y0, x1, y1 = inst.box
            instances.append(
                FeatureInstance(
                    cls=cls,
                    score=inst.score,
                    bbox=(round(x0), round(y0), round(x1), round(y1)),
                )
            )
    return FacadeFeatures(
        building_id=building_id,
        photo_id=winner.photo.photo_id,
        view_path=view_path,
        class_mask_paths=class_mask_paths,
        instances=instances,
    )


def _lookup_building(buildings: list[Building], building_id: str) -> Building:
    for b in buildings:
        if b.building_id == building_id:
            return b
    raise ValueError(f"no Building record for facade {building_id}")


@tag(stage="vision")
def facade_works(chosen_work_for_facade: Collect[FacadeWork]) -> dict[str, FacadeWork]:
    """Collect every facade's outcome keyed by building_id (audit records for
    all, mask+features only when status == "ok")."""
    return {w.audit.building_id: w for w in chosen_work_for_facade}


@tag(stage="vision")
def facade_masks(facade_works: dict[str, FacadeWork]) -> dict[str, FacadeMask]:
    return {bid: w.mask for bid, w in facade_works.items() if w.mask is not None}


@tag(stage="vision")
def facade_features(facade_works: dict[str, FacadeWork]) -> dict[str, FacadeFeatures]:
    return {bid: w.features for bid, w in facade_works.items() if w.features is not None}


def _evaluate_candidates(
    facade: Facade,
    building: Building,
    photos: list[PhotoMetadata],
    cfg: SamConfig,
    token: SecretStr,
) -> Iterable[_Candidate]:
    for rank, photo in enumerate(photos):
        try:
            view_bytes, bbox = _prepare_image(photo, facade, building, cfg, token)
        except httpx.HTTPError as exc:
            log.warning("Mapillary fetch failed for %s: %s", photo.photo_id, exc)
            continue
        facade_bbox_prompt = [[float(v) for v in bbox]] if bbox is not None else None
        # Facade + occluders + feature prompts ride one SAM call so the ViT
        # encodes each candidate image exactly once. Result slots line up
        # with the prompt list by index — keep them in this order.
        screening_prompts = [SamPrompt(cfg.facade_prompt, facade_bbox_prompt)] + [
            SamPrompt(p) for p in cfg.occluder_prompts
        ]
        feature_prompts = [SamPrompt(FEATURE_CLASS_PROMPT[p]) for p in cfg.feature_prompts]
        all_prompts = screening_prompts + feature_prompts
        try:
            results = sam_client.segment(
                view_bytes,
                prompts=all_prompts,
                base_url=str(cfg.service_url),
                timeout_s=cfg.http_timeout_s,
                retries=cfg.http_retries,
                retry_base_delay_s=cfg.http_retry_base_delay_s,
            )
        except httpx.HTTPError as exc:
            log.warning("sam-service call failed for %s: %s", photo.photo_id, exc)
            continue
        facade_instances = results[0]
        occluder_instances_lists = results[1 : 1 + len(cfg.occluder_prompts)]
        feature_instances_by_class = results[1 + len(cfg.occluder_prompts) :]
        if not facade_instances:
            continue
        facade_best = max(facade_instances, key=lambda inst: inst.score)
        if facade_best.score < cfg.min_facade_score:
            continue
        facade_bbox = _bbox_of_mask(facade_best.mask)
        if facade_bbox is None:
            continue
        if (
            bbox is not None
            and (overlap := _mask_bbox_overlap(facade_best.mask, bbox)) < cfg.min_mask_bbox_overlap
        ):
            log.warning(
                "SAM locked onto wrong region for %s (photo %s): "
                "mask-in-bbox=%.2f < %.2f; skipping candidate",
                facade.building_id,
                photo.photo_id,
                overlap,
                cfg.min_mask_bbox_overlap,
            )
            continue
        occluder_mask = _union_mask(
            [inst for group in occluder_instances_lists for inst in group],
            shape=facade_best.mask.shape,
        )
        yield _Candidate(
            photo=photo,
            rank=rank,
            view_bytes=view_bytes,
            facade_mask=facade_best.mask,
            facade_bbox=facade_bbox,
            occluder_mask=occluder_mask,
            facade_score=facade_best.score,
            occluder_ratio=_occluder_bbox_ratio(occluder_mask, facade_bbox),
            projected_bbox=bbox,
            feature_instances_by_class=feature_instances_by_class,
        )


def _prepare_image(
    photo: PhotoMetadata,
    facade: Facade,
    building: Building,
    cfg: SamConfig,
    token: SecretStr,
) -> tuple[bytes, tuple[int, int, int, int] | None]:
    """Pano: hi-res fetch + reproject + project bbox. Perspective: raw thumb, no bbox."""
    if not photo.is_pano:
        raw = mapillary.fetch_image_bytes(photo.url, timeout_s=cfg.mapillary_image_timeout_s)
        return raw, None
    hires_url = mapillary.fetch_thumb_url(photo.photo_id, field=cfg.pano_thumb_field, token=token)
    raw_bytes = mapillary.fetch_image_bytes(hires_url, timeout_s=cfg.mapillary_image_timeout_s)
    pitch_deg = _view_pitch_deg(photo, facade, building, cfg)
    view_bytes = rectilinear_view(
        raw_bytes,
        yaw_deg=_target_bearing(photo, facade),
        pano_yaw_deg=photo.bearing_deg if photo.bearing_deg is not None else 0.0,
        pitch_deg=pitch_deg,
        fov_deg=cfg.pano_view_fov_deg,
        size_px=cfg.pano_view_size,
    )
    bbox = _project_footprint_bbox(facade, building, photo, cfg, pitch_deg)
    return view_bytes, bbox


def _view_pitch_deg(
    photo: PhotoMetadata, facade: Facade, building: Building, cfg: SamConfig
) -> float:
    """Pitch so the roofline sits just inside the top of the view.

    Aiming at the facade mid-height (naive) crops the top for tall close
    buildings. Aiming so the roof is ``_sky_margin_deg`` below the view's
    top edge guarantees the gable is captured (and a strip of sky above it,
    which is what the user sees to confirm 'whole building in frame').
    Ground floor is cropped for close tall buildings — unavoidable at a
    45° FoV, but door detection is a lesser concern than roofline for
    canal-house grammars.
    """
    camera_z, ground_z = _vertical_reference(photo, building)
    roof_z = ground_z + building.height_m
    lon, lat = facade.midpoint
    _, _, distance_m = WGS84_GEOD.inv(photo.lon, photo.lat, lon, lat)
    if distance_m < 1e-6:
        return 0.0
    roof_elev_deg = math.degrees(math.atan2(roof_z - camera_z, distance_m))
    w, h = cfg.pano_view_size
    fov_rad = math.radians(cfg.pano_view_fov_deg)
    focal = w / (2 * math.tan(fov_rad / 2))
    vert_half_deg = math.degrees(math.atan2(h / 2, focal))
    return max(0.0, roof_elev_deg - vert_half_deg + _SKY_MARGIN_DEG)


def _target_bearing(photo: PhotoMetadata, facade: Facade) -> float:
    lon, lat = facade.midpoint
    azimuth, _, _ = WGS84_GEOD.inv(photo.lon, photo.lat, lon, lat)
    return azimuth % 360.0


_FALLBACK_CAMERA_HEIGHT_M: Final = 2.5
# Degrees of sky padding above the roofline target. Tuned so canal-house
# gables sit a visible strip below the top of the reprojected view;
# changing it requires FG_CACHE__RECOMPUTE_NODES=chosen_work_for_facade
# because it's not on SamConfig (would bust the 500-building cache).
_SKY_MARGIN_DEG: Final = 3.0


def _vertical_reference(photo: PhotoMetadata, building: Building) -> tuple[float, float]:
    """Return ``(camera_z, ground_z)`` in a consistent vertical frame.

    Mapillary's EGM96 altitude is within ~1 m of 3D BAG's NAP ground in NL, so
    we use both directly when available; otherwise fall back to camera-2.5m.
    """
    if photo.altitude_m is not None:
        return photo.altitude_m, building.ground_altitude_m
    return _FALLBACK_CAMERA_HEIGHT_M, 0.0


def _project_footprint_bbox(
    facade: Facade,
    building: Building,
    photo: PhotoMetadata,
    cfg: SamConfig,
    view_pitch_deg: float,
) -> tuple[int, int, int, int] | None:
    """Pixel bbox of the canal-facing edge at ground + roof peak, with safety margin.

    We used to project the *full* footprint, but for deep narrow canal houses
    (2-3m wide, 15m deep) that bounded the back wall too, widening the bbox
    well past the actual visible facade. A mask sitting on a neighbour's wall
    could still overlap our loose bbox and pass the sanity check. Using just
    the facade edge's two endpoints gives a tight bound on exactly what SAM
    should be looking at. ``view_pitch_deg`` must match the pitch passed to
    ``rectilinear_view`` or the bbox will be vertically offset.
    """
    target_bearing = _target_bearing(photo, facade)
    camera_z, ground_z = _vertical_reference(photo, building)
    heights = (ground_z, ground_z + building.height_m)
    edges = (facade.edge_start, facade.edge_end)
    projected: list[tuple[float, float]] = []
    for (lon, lat), z in itertools.product(edges, heights):
        pt = world_to_pixel(
            photo_lon=photo.lon,
            photo_lat=photo.lat,
            camera_z_m=camera_z,
            world_lon=lon,
            world_lat=lat,
            world_z_m=z,
            view_yaw_deg=target_bearing,
            view_pitch_deg=view_pitch_deg,
            fov_deg=cfg.pano_view_fov_deg,
            size_px=cfg.pano_view_size,
        )
        if pt is not None:
            projected.append(pt)
    if len(projected) < 2:
        return None
    xs = [p[0] for p in projected]
    ys = [p[1] for p in projected]
    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    margin_x = (x1 - x0) * cfg.bbox_margin_frac
    margin_y = (y1 - y0) * cfg.bbox_margin_frac
    view_w, view_h = cfg.pano_view_size
    return (
        max(0, int(x0 - margin_x)),
        max(0, int(y0 - margin_y)),
        min(view_w, int(x1 + margin_x)),
        min(view_h, int(y1 + margin_y)),
    )


def _occluder_bbox_ratio(occluder_mask: BoolMask, facade_bbox: tuple[int, int, int, int]) -> float:
    """Fraction of the facade's bounding rectangle occupied by occluder pixels."""
    x0, y0, x1, y1 = facade_bbox
    return float(occluder_mask[y0:y1, x0:x1].sum()) / ((y1 - y0) * (x1 - x0))


def _pick_best(candidates: list[_Candidate], max_occluder_ratio: float) -> _Candidate | None:
    """Hard-filter heavily occluded candidates, then pick by selection rank.

    Rank is already a perpendicularity/winter/quality-weighted ordering from
    ``top_photos_per_facade``, so 'pick min rank among survivors' preserves
    head-on preference while refusing to reward an oblique-but-clean shot
    over a mostly-clean head-on one. If every candidate is above the ratio,
    we still return the best-ranked rather than drop the facade.
    """
    if not candidates:
        return None
    clean = [c for c in candidates if c.occluder_ratio <= max_occluder_ratio]
    pool = clean if clean else candidates
    return min(pool, key=lambda c: c.rank)


def _write_mask(mask: BoolMask, path: Path) -> None:
    Image.fromarray((mask * 255).astype(np.uint8)).convert("1").save(
        path, format="PNG", optimize=True
    )


def _bbox_of_mask(mask: BoolMask) -> tuple[int, int, int, int] | None:
    """Axis-aligned xyxy bbox of True pixels, or None if the mask is empty."""
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _mask_bbox_overlap(mask: BoolMask, bbox: tuple[int, int, int, int]) -> float:
    """Fraction of the mask's True pixels that fall inside ``bbox``.

    Cheap wrong-target detector: if SAM's 'building' mask sits mostly outside
    the region we projected the target footprint into, it's a neighbour or
    a tree the segmenter mis-labelled, not our target.
    """
    total = int(mask.sum())
    if total == 0:
        return 0.0
    x0, y0, x1, y1 = bbox
    return int(mask[y0:y1, x0:x1].sum()) / total


def _centre_inside(
    bbox: tuple[float, float, float, float],
    container: tuple[int, int, int, int],
) -> bool:
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    return container[0] <= cx <= container[2] and container[1] <= cy <= container[3]


def _filter_instances(
    instances: list[SamInstance],
    container: tuple[int, int, int, int],
    min_score: float,
) -> list[SamInstance]:
    return [
        inst
        for inst in instances
        if inst.score >= min_score and _centre_inside(inst.box, container)
    ]


def _union_mask(instances: list[SamInstance], *, shape: tuple[int, ...]) -> BoolMask:
    if not instances:
        return np.zeros(shape, dtype=bool)
    return np.logical_or.reduce([inst.mask for inst in instances])
