"""Photo-to-facade matching: hard filters, then across-canal preference + distance rank."""

import math

import polars as pl
from hamilton.function_modifiers import tag
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree

from facade_grammar.config import SelectionConfig
from facade_grammar.geo import WGS84_TO_RD, build_rd_geoms
from facade_grammar.schemas.buildings import Facade
from facade_grammar.schemas.photos import PhotoMetadata


@tag(stage="selection")
def top_photos_per_facade(
    canal_facades: list[Facade],
    raw_photo_metadata: list[PhotoMetadata],
    raw_waterways: pl.DataFrame,
    selection: SelectionConfig,
) -> dict[str, list[PhotoMetadata]]:
    """For each canal facade, up to ``top_k`` candidate photos ordered best-first.

    Across-canal photos (sight line from photo to facade midpoint intersects a
    waterway geometry) sort ahead of same-side photos; within each bucket the
    closer photo wins. Facades with no surviving candidates produce no key.
    """
    photos_rd = _reproject_photos(raw_photo_metadata)
    waterway_tree = STRtree(build_rd_geoms(raw_waterways["geometry_wkt"].to_list()))
    return {
        facade.building_id: candidates
        for facade, (edge, midpoint) in zip(
            canal_facades, _batch_facade_geoms(canal_facades), strict=True
        )
        if (
            candidates := _top_k_photos(
                facade, edge, midpoint, raw_photo_metadata, photos_rd, waterway_tree, selection
            )
        )
    }


def _reproject_photos(photos: list[PhotoMetadata]) -> list[tuple[float, float]]:
    if not photos:
        return []
    lons = [p.lon for p in photos]
    lats = [p.lat for p in photos]
    xs, ys = WGS84_TO_RD.transform(lons, lats)
    return list(zip(xs, ys, strict=True))


def _batch_facade_geoms(
    facades: list[Facade],
) -> list[tuple[LineString, tuple[float, float]]]:
    """Reproject every facade's endpoints to RD in two batched pyproj calls."""
    if not facades:
        return []
    start_xs, start_ys = WGS84_TO_RD.transform(
        [f.edge_start[0] for f in facades], [f.edge_start[1] for f in facades]
    )
    end_xs, end_ys = WGS84_TO_RD.transform(
        [f.edge_end[0] for f in facades], [f.edge_end[1] for f in facades]
    )
    return [
        (LineString([(sx, sy), (ex, ey)]), ((sx + ex) / 2, (sy + ey) / 2))
        for sx, sy, ex, ey in zip(start_xs, start_ys, end_xs, end_ys, strict=True)
    ]


def _top_k_photos(
    facade: Facade,
    edge: LineString,
    midpoint: tuple[float, float],
    photos: list[PhotoMetadata],
    photos_rd: list[tuple[float, float]],
    waterway_tree: STRtree,
    selection: SelectionConfig,
) -> list[PhotoMetadata]:
    normal = (
        math.sin(math.radians(facade.normal_deg)),
        math.cos(math.radians(facade.normal_deg)),
    )
    # Rank by (same_side, distance): across-canal photos have same_side=False,
    # which sorts ahead of True; within each bucket the closer photo wins.
    ranked = sorted(
        (
            (
                not _crosses_water(LineString([rd, midpoint]), waterway_tree),
                edge.distance(Point(rd)),
                photo,
            )
            for photo, rd in zip(photos, photos_rd, strict=True)
            if _passes_hard_filters(
                photo, rd, edge, midpoint, normal, facade.normal_deg, selection
            )
        ),
        key=lambda t: (t[0], t[1]),
    )
    return [photo for _, _, photo in ranked[: selection.top_k]]


def _crosses_water(sight_line: LineString, waterway_tree: STRtree) -> bool:
    return waterway_tree.query(sight_line, predicate="intersects").size > 0


def _passes_hard_filters(
    photo: PhotoMetadata,
    photo_rd: tuple[float, float],
    edge: LineString,
    midpoint: tuple[float, float],
    normal: tuple[float, float],
    normal_deg: float,
    selection: SelectionConfig,
) -> bool:
    px, py = photo_rd
    mx, my = midpoint
    if abs(px - mx) > selection.photo_max_dist_m or abs(py - my) > selection.photo_max_dist_m:
        return False
    dist = edge.distance(Point(px, py))
    if not (selection.photo_min_dist_m <= dist <= selection.photo_max_dist_m):
        return False
    # outside-of-facade: photo must be on the outward half-plane
    if (px - mx) * normal[0] + (py - my) * normal[1] <= 0:
        return False
    # Panoramas see 360 degrees; we'll reproject toward the facade at vision-time.
    if photo.is_pano:
        return True
    if photo.bearing_deg is None:
        return False
    # bearing alignment: the perspective photo should look back toward the facade
    delta = ((photo.bearing_deg - (normal_deg + 180)) + 540) % 360 - 180
    return abs(delta) <= selection.bearing_tol_deg
