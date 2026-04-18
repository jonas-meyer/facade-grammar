"""Tests for pure-logic helpers in ``facade_grammar.pipeline.ingestion``."""

from facade_grammar.config import Bbox
from facade_grammar.pipeline.ingestion import _expand_bbox


def test_expand_bbox_grows_in_all_directions() -> None:
    original = Bbox(4.88, 52.37, 4.89, 52.38)
    expanded = _expand_bbox(original, buffer_m=50.0)
    assert expanded.min_lon < original.min_lon
    assert expanded.min_lat < original.min_lat
    assert expanded.max_lon > original.max_lon
    assert expanded.max_lat > original.max_lat


def test_expand_bbox_buffer_is_approximately_metres() -> None:
    # At 52.37°N, 100 m of latitude is ~9.0e-4 deg; 100 m of longitude is
    # ~100 / (111_320 * cos(52.37°)) ≈ 1.47e-3 deg. Allow a generous range
    # to tolerate the RD-reprojection path's slight curvature.
    original = Bbox(4.88, 52.37, 4.89, 52.38)
    expanded = _expand_bbox(original, buffer_m=100.0)
    lat_delta = expanded.max_lat - original.max_lat
    lon_delta = expanded.max_lon - original.max_lon
    assert 7e-4 < lat_delta < 11e-4
    assert 11e-4 < lon_delta < 18e-4
