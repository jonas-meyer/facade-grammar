"""Per-facade SAM 3 segmentation: pick the best candidate photo and persist its mask.

Panos get reprojected to a rectilinear view facing the target facade; the
projected footprint bbox is passed to SAM as a visual prompt so it locks
onto our target house in a multi-house scene.
"""

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import NamedTuple

import httpx
import numpy as np
from hamilton.function_modifiers import tag
from hamilton.htypes import Collect, Parallelizable
from PIL import Image
from pydantic import SecretStr

from facade_grammar.clients import mapillary
from facade_grammar.clients import sam as sam_client
from facade_grammar.clients.sam import SamInstance, SamPrompt
from facade_grammar.config import SamConfig
from facade_grammar.geo import WGS84_GEOD
from facade_grammar.schemas.buildings import (
    Building,
    Facade,
    FacadeFeatures,
    FacadeMask,
    FeatureClass,
    FeatureInstance,
)
from facade_grammar.schemas.photos import PhotoMetadata
from facade_grammar.viz.panorama import rectilinear_view, world_to_pixel

log = logging.getLogger(__name__)


class _Candidate(NamedTuple):
    photo: PhotoMetadata
    rank: int
    view_bytes: bytes
    facade_mask: np.ndarray
    occluder_mask: np.ndarray
    facade_score: float
    occluder_ratio: float
    projected_bbox: tuple[int, int, int, int] | None


@tag(stage="vision")
def per_facade_photos(
    top_photos_per_facade: dict[str, list[PhotoMetadata]],
    canal_facades: list[Facade],
    sam: SamConfig,
) -> Parallelizable[tuple[Facade, list[PhotoMetadata]]]:
    """Fan out: one parallel task per canal-front facade, optionally capped for testing."""
    facades_by_bid = {f.building_id: f for f in canal_facades}
    items = [
        (facades_by_bid[bid], photos) for bid, photos in top_photos_per_facade.items()
    ]
    if sam.max_buildings is not None:
        items = items[: sam.max_buildings]
    yield from items


@tag(stage="vision")
def chosen_mask_for_facade(
    per_facade_photos: tuple[Facade, list[PhotoMetadata]],
    raw_buildings: list[Building],
    sam: SamConfig,
    mapillary_token: SecretStr,
) -> FacadeMask | None:
    """Run SAM 3 against the top-K candidates for one facade; persist the winner's masks."""
    facade, photos = per_facade_photos
    building = _lookup_building(raw_buildings, facade.building_id)
    best = _pick_best(list(_evaluate_candidates(facade, building, photos, sam, mapillary_token)))
    if best is None:
        return None
    sam.mask_dir.mkdir(parents=True, exist_ok=True)
    view_path = sam.mask_dir / f"{facade.building_id}_view.jpg"
    facade_path = sam.mask_dir / f"{facade.building_id}_facade.png"
    occluder_path = sam.mask_dir / f"{facade.building_id}_occluder.png"
    view_path.write_bytes(best.view_bytes)
    _write_mask(best.facade_mask, facade_path)
    _write_mask(best.occluder_mask, occluder_path)
    return FacadeMask(
        building_id=facade.building_id,
        photo_id=best.photo.photo_id,
        view_path=view_path,
        mask_path=facade_path,
        occluder_mask_path=occluder_path,
        facade_score=best.facade_score,
        occluder_ratio=best.occluder_ratio,
        projected_bbox=best.projected_bbox,
    )


def _lookup_building(buildings: list[Building], building_id: str) -> Building:
    for b in buildings:
        if b.building_id == building_id:
            return b
    raise ValueError(f"no Building record for facade {building_id}")


@tag(stage="vision")
def facade_masks(chosen_mask_for_facade: Collect[FacadeMask | None]) -> dict[str, FacadeMask]:
    """Collect per-facade results, dropping facades with no survivor."""
    return {m.building_id: m for m in chosen_mask_for_facade if m is not None}


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
        all_prompts = [SamPrompt(cfg.facade_prompt, facade_bbox_prompt)] + [
            SamPrompt(p) for p in cfg.occluder_prompts
        ]
        results = sam_client.segment(
            view_bytes,
            prompts=all_prompts,
            base_url=str(cfg.service_url),
            timeout_s=cfg.http_timeout_s,
        )
        facade_instances, *occluder_instances_lists = results
        if not facade_instances:
            continue
        facade_best = max(facade_instances, key=lambda inst: inst.score)
        if facade_best.score < cfg.min_facade_score:
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
            occluder_mask=occluder_mask,
            facade_score=facade_best.score,
            occluder_ratio=_occluder_bbox_ratio(facade_best.mask, occluder_mask),
            projected_bbox=bbox,
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
    hires_url = mapillary.fetch_thumb_url(
        photo.photo_id, field=cfg.pano_thumb_field, token=token
    )
    raw_bytes = mapillary.fetch_image_bytes(hires_url, timeout_s=cfg.mapillary_image_timeout_s)
    view_bytes = rectilinear_view(
        raw_bytes,
        yaw_deg=_target_bearing(photo, facade),
        pano_yaw_deg=photo.bearing_deg if photo.bearing_deg is not None else 0.0,
        fov_deg=cfg.pano_view_fov_deg,
        size_px=cfg.pano_view_size,
    )
    bbox = _project_footprint_bbox(facade, building, photo, cfg)
    return view_bytes, bbox


def _target_bearing(photo: PhotoMetadata, facade: Facade) -> float:
    lon, lat = facade.midpoint
    azimuth, _, _ = WGS84_GEOD.inv(photo.lon, photo.lat, lon, lat)
    return azimuth % 360.0


_FALLBACK_CAMERA_HEIGHT_M = 2.5


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
) -> tuple[int, int, int, int] | None:
    """Pixel bbox of the full footprint at ground + roof peak, with safety margin."""
    target_bearing = _target_bearing(photo, facade)
    camera_z, ground_z = _vertical_reference(photo, building)
    heights = (ground_z, ground_z + building.height_m)
    projected: list[tuple[float, float]] = []
    for lon, lat in building.footprint:
        for z in heights:
            pt = world_to_pixel(
                photo_lon=photo.lon,
                photo_lat=photo.lat,
                camera_z_m=camera_z,
                world_lon=lon,
                world_lat=lat,
                world_z_m=z,
                view_yaw_deg=target_bearing,
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


def _occluder_bbox_ratio(facade_mask: np.ndarray, occluder_mask: np.ndarray) -> float:
    """Fraction of the facade's bounding rectangle occupied by occluder pixels."""
    bbox = _bbox_of_mask(facade_mask)
    if bbox is None:
        return 0.0
    x0, y0, x1, y1 = bbox
    bbox_area = (y1 - y0) * (x1 - x0)
    if bbox_area == 0:
        return 0.0
    return float(occluder_mask[y0:y1, x0:x1].sum()) / bbox_area


def _pick_best(candidates: list[_Candidate]) -> _Candidate | None:
    if not candidates:
        return None
    return min(candidates, key=lambda c: (c.occluder_ratio, c.rank))


def _write_mask(mask: np.ndarray, path: Path) -> None:
    Image.fromarray((mask * 255).astype(np.uint8)).convert("1").save(
        path, format="PNG", optimize=True
    )


@tag(stage="vision")
def per_facade_mask_for_features(
    facade_masks: dict[str, FacadeMask],
) -> Parallelizable[FacadeMask]:
    """Fan out: one parallel task per facade that survived Phase 4."""
    yield from facade_masks.values()


@tag(stage="vision")
def features_for_facade(
    per_facade_mask_for_features: FacadeMask,
    sam: SamConfig,
) -> FacadeFeatures | None:
    """Run SAM against the Phase 4 view to segment sub-features (window/door/gable/floor)."""
    fm = per_facade_mask_for_features
    view_bytes = fm.view_path.read_bytes()
    facade_mask = np.asarray(Image.open(fm.mask_path), dtype=bool)
    facade_bbox = _bbox_of_mask(facade_mask)
    if facade_bbox is None:
        return None

    prompts = [
        SamPrompt(cls, [[float(v) for v in facade_bbox]]) for cls in sam.feature_prompts
    ]
    per_class_results = sam_client.segment(
        view_bytes,
        prompts=prompts,
        base_url=str(sam.service_url),
        timeout_s=sam.http_timeout_s,
    )

    out_dir = sam.features_dir / fm.building_id
    out_dir.mkdir(parents=True, exist_ok=True)

    instances: list[FeatureInstance] = []
    class_mask_paths: dict[FeatureClass, Path] = {}
    for cls, class_instances in zip(sam.feature_prompts, per_class_results, strict=True):
        kept = _filter_instances(class_instances, facade_bbox, sam.feature_min_score)
        class_mask_path = out_dir / f"{cls}.png"
        _write_mask(_union_mask(kept, shape=facade_mask.shape), class_mask_path)
        class_mask_paths[cls] = class_mask_path
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
        building_id=fm.building_id,
        photo_id=fm.photo_id,
        view_path=fm.view_path,
        class_mask_paths=class_mask_paths,
        instances=instances,
    )


@tag(stage="vision")
def facade_features(
    features_for_facade: Collect[FacadeFeatures | None],
) -> dict[str, FacadeFeatures]:
    """Collect per-facade features, dropping facades with no survivor."""
    return {f.building_id: f for f in features_for_facade if f is not None}


def _bbox_of_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Axis-aligned xyxy bbox of True pixels, or None if the mask is empty."""
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


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


def _union_mask(instances: list[SamInstance], *, shape: tuple[int, ...]) -> np.ndarray:
    if not instances:
        return np.zeros(shape, dtype=bool)
    return np.logical_or.reduce([inst.mask for inst in instances])
