"""Tests for the pure-logic helpers in ``facade_grammar.pipeline.spatial``."""

import math

from shapely.geometry import LineString, Point
from shapely.strtree import STRtree

from facade_grammar.pipeline.spatial import (
    _edge_normal_deg,
    _ensure_ccw,
    _facade_sort_key,
    _faces_feature,
    _FeatureIndex,
    _longest_canal_facade,
)
from facade_grammar.schemas.buildings import Facade

# Unit square in RD, CCW vertex order: (0,0)->(1,0)->(1,1)->(0,1)->(0,0).
_CCW = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)]
_CW = list(reversed(_CCW))


def test_ensure_ccw_passes_ccw_unchanged() -> None:
    assert _ensure_ccw(_CCW) == _CCW


def test_ensure_ccw_reverses_cw_ring() -> None:
    assert _ensure_ccw(_CW) == _CCW


def test_edge_normal_deg_cardinal_edges_of_ccw_square() -> None:
    assert _edge_normal_deg(_CCW[0], _CCW[1]) == 180.0  # south edge, outward = south
    assert _edge_normal_deg(_CCW[1], _CCW[2]) == 90.0  # east edge
    assert _edge_normal_deg(_CCW[2], _CCW[3]) == 0.0  # north edge
    assert _edge_normal_deg(_CCW[3], _CCW[0]) == 270.0  # west edge


def test_edge_normal_deg_diagonal() -> None:
    # Edge pointing NE: tangent (1,1), outward normal is -90deg rotation = (1,-1),
    # compass azimuth = atan2(1, -1) = 135 deg (SE).
    normal = _edge_normal_deg((0.0, 0.0), (1.0, 1.0))
    assert math.isclose(normal, 135.0, abs_tol=1e-9)


def _facade(building_id: str, start: tuple[float, float], end: tuple[float, float]) -> Facade:
    return Facade(
        building_id=building_id,
        edge_start=start,
        edge_end=end,
        classification="canal",
        normal_deg=0.0,
    )


def test_longest_canal_facade_picks_longest_by_length() -> None:
    short = _facade("b1", (0.0, 0.0), (1.0, 0.0))
    long = _facade("b1", (0.0, 0.0), (3.0, 0.0))
    assert _longest_canal_facade([short, long]) is long
    assert _longest_canal_facade([long, short]) is long  # order-independent


def test_longest_canal_facade_tiebreak_on_edge_start() -> None:
    a = _facade("b1", (0.0, 0.0), (1.0, 0.0))
    b = _facade("b1", (5.0, 5.0), (6.0, 5.0))
    # Equal length, max() picks the tuple with larger edge_start lexicographically.
    assert _longest_canal_facade([a, b]) is b


def test_facade_sort_key_shape() -> None:
    f = _facade("b", (1.0, 2.0), (4.0, 6.0))
    length_sq, start = _facade_sort_key(f)
    assert length_sq == 9.0 + 16.0
    assert start == (1.0, 2.0)


# --- _faces_feature ------------------------------------------------------
#
# Synthetic fixture: facade midpoint at the origin, outward normal pointing
# north (0, 1). Water geometries are positioned to test each rejection path.

_NORTH_NORMAL = (0.0, 1.0)
_MID = Point(0.0, 0.0)


def _features(*geoms: LineString) -> _FeatureIndex:
    gs = list(geoms)
    return _FeatureIndex(STRtree(gs), gs)  # type: ignore[arg-type]


def test_faces_feature_accepts_water_in_forward_cone() -> None:
    features = _features(LineString([(-100.0, 10.0), (100.0, 10.0)]))
    assert _faces_feature(_MID, _NORTH_NORMAL, features, max_distance_m=15.0, cos_min=0.5)


def test_faces_feature_rejects_water_behind_facade() -> None:
    features = _features(LineString([(-100.0, -10.0), (100.0, -10.0)]))
    assert not _faces_feature(_MID, _NORTH_NORMAL, features, max_distance_m=15.0, cos_min=0.5)


def test_faces_feature_rejects_water_beyond_max_distance() -> None:
    features = _features(LineString([(-100.0, 20.0), (100.0, 20.0)]))
    assert not _faces_feature(_MID, _NORTH_NORMAL, features, max_distance_m=15.0, cos_min=0.5)


def test_faces_feature_rejects_water_perpendicular_to_normal() -> None:
    # Nearest point on an east-side strip is directly east of midpoint;
    # cos(angle between north-normal and east) = 0 < cos_min.
    features = _features(LineString([(10.0, -100.0), (10.0, 100.0)]))
    assert not _faces_feature(_MID, _NORTH_NORMAL, features, max_distance_m=15.0, cos_min=0.5)
