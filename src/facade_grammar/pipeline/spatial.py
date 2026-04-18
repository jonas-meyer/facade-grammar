"""Spatial classification of footprint edges (canal / street / other)."""

import math
from itertools import pairwise
from typing import NamedTuple

import polars as pl
import shapely.ops
from hamilton.function_modifiers import tag
from shapely.geometry import LineString, Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.strtree import STRtree

from facade_grammar.config import SpatialConfig
from facade_grammar.geo import WGS84_TO_RD, build_rd_geoms
from facade_grammar.schemas.buildings import Building, Facade, FacadeClass


class _FeatureIndex(NamedTuple):
    """STRtree paired with the reprojected geoms it was built from."""

    tree: STRtree
    geoms: list[BaseGeometry]


def _index_from_wkts(wkts: list[str]) -> _FeatureIndex:
    geoms = build_rd_geoms(wkts)
    return _FeatureIndex(STRtree(geoms), geoms)


@tag(stage="spatial")
def classified_facades(
    raw_buildings: list[Building],
    raw_streets: pl.DataFrame,
    raw_waterways: pl.DataFrame,
    spatial: SpatialConfig,
) -> list[Facade]:
    """One ``Facade`` per footprint edge, classified canal / street / other.

    An edge classifies as ``canal`` when its outward normal points toward a waterway
    within ``canal_near_m``; same logic with ``street_near_m`` for ``street``. The
    direction check (``forward_cos_min``) prevents side walls of narrow canal houses
    from inheriting their neighbour's canal proximity. Geometric math happens in
    EPSG:28992 (metres); edge coordinates are reported back in WGS84 (reusing the
    building's original footprint, no reprojection roundtrip).
    """
    waterways = _index_from_wkts(raw_waterways["geometry_wkt"].to_list())
    streets = _index_from_wkts(raw_streets["geometry_wkt"].to_list())

    facades: list[Facade] = []
    for building in raw_buildings:
        ring_wgs = _ensure_ccw(building.footprint)
        ring_rd = _reproject_ring(ring_wgs)
        facades.extend(
            _classify_facade(building.building_id, wgs_edge, rd_edge, waterways, streets, spatial)
            for wgs_edge, rd_edge in zip(pairwise(ring_wgs), pairwise(ring_rd), strict=True)
        )
    return facades


@tag(stage="spatial")
def canal_facades(classified_facades: list[Facade]) -> list[Facade]:
    """The single longest canal-classified edge per building.

    Narrow canal houses often have a short front and two long party walls with
    canal water behind them; the front can also be one of several canal-classified
    edges. Picking the longest canal-classified edge per building keeps one
    representative facade per canal-front address.
    """
    by_building: dict[str, list[Facade]] = {}
    for f in classified_facades:
        if f.classification == "canal":
            by_building.setdefault(f.building_id, []).append(f)
    return [_longest_canal_facade(group) for group in by_building.values()]


def _classify_facade(
    building_id: str,
    wgs_edge: tuple[tuple[float, float], tuple[float, float]],
    rd_edge: tuple[tuple[float, float], tuple[float, float]],
    waterways: _FeatureIndex,
    streets: _FeatureIndex,
    spatial: SpatialConfig,
) -> Facade:
    rd_start, rd_end = rd_edge
    normal_deg = _edge_normal_deg(rd_start, rd_end)
    return Facade(
        building_id=building_id,
        edge_start=wgs_edge[0],
        edge_end=wgs_edge[1],
        classification=_classify_edge(
            LineString([rd_start, rd_end]), normal_deg, waterways, streets, spatial
        ),
        normal_deg=normal_deg,
    )


def _reproject_ring(footprint_wgs84: list[tuple[float, float]]) -> list[tuple[float, float]]:
    lons = [p[0] for p in footprint_wgs84]
    lats = [p[1] for p in footprint_wgs84]
    xs, ys = WGS84_TO_RD.transform(lons, lats)
    return list(zip(xs, ys, strict=True))


def _ensure_ccw(ring_rd: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if Polygon(ring_rd).exterior.is_ccw:
        return ring_rd
    return list(reversed(ring_rd))


def _edge_normal_deg(p0: tuple[float, float], p1: tuple[float, float]) -> float:
    """Outward-normal compass azimuth for an edge of a CCW-wound polygon (0=N, 90=E)."""
    tx = p1[0] - p0[0]
    ty = p1[1] - p0[1]
    # For a CCW ring the outward normal is the tangent rotated -90 degrees.
    nx = ty
    ny = -tx
    return math.degrees(math.atan2(nx, ny)) % 360


def _classify_edge(
    edge_rd: LineString,
    normal_deg: float,
    waterways: _FeatureIndex,
    streets: _FeatureIndex,
    spatial: SpatialConfig,
) -> FacadeClass:
    mid = edge_rd.interpolate(0.5, normalized=True)
    normal = (math.sin(math.radians(normal_deg)), math.cos(math.radians(normal_deg)))
    if _faces_feature(mid, normal, waterways, spatial.canal_near_m, spatial.forward_cos_min):
        return "canal"
    if _faces_feature(mid, normal, streets, spatial.street_near_m, spatial.forward_cos_min):
        return "street"
    return "other"


def _faces_feature(
    mid: Point,
    normal: tuple[float, float],
    features: _FeatureIndex,
    max_distance_m: float,
    cos_min: float,
) -> bool:
    """True if any feature within ``max_distance_m`` lies in the outward-normal cone."""
    for idx in features.tree.query(mid, predicate="dwithin", distance=max_distance_m):
        nearest = shapely.ops.nearest_points(mid, features.geoms[idx])[1]
        dx, dy = nearest.x - mid.x, nearest.y - mid.y
        mag = math.hypot(dx, dy)
        if mag < 1e-6:
            return True
        cos_angle = (dx * normal[0] + dy * normal[1]) / mag
        if cos_angle >= cos_min:
            return True
    return False


def _longest_canal_facade(group: list[Facade]) -> Facade:
    return max(group, key=_facade_sort_key)


def _facade_sort_key(f: Facade) -> tuple[float, tuple[float, float]]:
    dx = f.edge_end[0] - f.edge_start[0]
    dy = f.edge_end[1] - f.edge_start[1]
    return (dx * dx + dy * dy, f.edge_start)
