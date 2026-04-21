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
    photos_rd = reproject_photos(raw_photo_metadata)
    waterway_tree = STRtree(build_rd_geoms(raw_waterways["geometry_wkt"].to_list()))
    return {
        facade.building_id: candidates
        for facade, (edge, midpoint) in zip(
            canal_facades, batch_facade_geoms(canal_facades), strict=True
        )
        if (
            candidates := _top_k_photos(
                facade, edge, midpoint, raw_photo_metadata, photos_rd, waterway_tree, selection
            )
        )
    }


def reproject_photos(photos: list[PhotoMetadata]) -> list[tuple[float, float]]:
    """Batch-project photo lon/lat to EPSG:28992 metres (shared with the audit stage)."""
    if not photos:
        return []
    lons = [p.lon for p in photos]
    lats = [p.lat for p in photos]
    xs, ys = WGS84_TO_RD.transform(lons, lats)
    return list(zip(xs, ys, strict=True))


def batch_facade_geoms(
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


def perpendicularity_deg(
    facade_normal_deg: float,
    midpoint: tuple[float, float],
    photo_rd: tuple[float, float],
) -> float:
    """|angular offset between photo→midpoint direction and the facade's
    inward normal|, in degrees. 0 = head-on, 90 = alongside the facade.

    Computed in RD (EPSG:28992) metres so the atan2 is in metric space — at
    Amsterdam latitudes the lon/lat distortion is small, but this stays
    consistent with everything else in selection.
    """
    mx, my = midpoint
    px, py = photo_rd
    bearing = math.degrees(math.atan2(mx - px, my - py)) % 360
    inward = (facade_normal_deg + 180) % 360
    delta = (bearing - inward + 540) % 360 - 180
    return abs(delta)


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
    # Sort key (across-canal, -year, -winter, perpendicularity, -quality, distance):
    # newer captures beat older because Mapillary's 2018-era imagery is
    # materially lower quality — so a 2024 pano wins over a 2020 winter
    # shot. Winter is a same-year tiebreaker for leafless views.
    ranked = sorted(
        (
            (
                not _crosses_water(LineString([rd, midpoint]), waterway_tree),
                -photo.captured_at.year,
                -int(photo.captured_at.month in selection.winter_months),
                perpendicularity_deg(facade.normal_deg, midpoint, rd),
                -(photo.quality_score if photo.quality_score is not None else 0.0),
                edge.distance(Point(rd)),
                photo,
            )
            for photo, rd in zip(photos, photos_rd, strict=True)
            if _passes_hard_filters(photo, rd, edge, midpoint, normal, facade.normal_deg, selection)
        ),
        key=lambda t: (t[0], t[1], t[2], t[3], t[4], t[5]),
    )
    return [photo for *_, photo in ranked[: selection.top_k]]


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
    if not photo.is_pano and dist > selection.perspective_max_dist_m:
        return False
    # Photo must sit on the outward half-plane of the facade edge.
    if (px - mx) * normal[0] + (py - my) * normal[1] <= 0:
        return False
    if (
        selection.min_quality_score is not None
        and photo.quality_score is not None
        and photo.quality_score < selection.min_quality_score
    ):
        return False
    if perpendicularity_deg(normal_deg, midpoint, photo_rd) > selection.max_perpendicularity_deg:
        return False
    if selection.winter_only and photo.captured_at.month not in selection.winter_months:
        return False
    if selection.pano_only and not photo.is_pano:
        return False
    # Panos get reprojected toward the facade at vision-time, so the bearing
    # check below — which only applies to fixed-FoV perspective photos —
    # isn't meaningful for them.
    if photo.is_pano:
        return True
    if photo.bearing_deg is None:
        return False
    delta = ((photo.bearing_deg - (normal_deg + 180)) + 540) % 360 - 180
    return abs(delta) <= selection.bearing_tol_deg
