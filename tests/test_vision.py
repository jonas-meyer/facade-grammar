"""Tests for pure helpers in ``facade_grammar.pipeline.vision``."""

from datetime import UTC, datetime

import numpy as np
from pydantic import HttpUrl

from facade_grammar.clients.sam import SamInstance
from facade_grammar.config import SamConfig
from facade_grammar.pipeline.vision import (
    _bbox_of_mask,
    _Candidate,
    _centre_inside,
    _filter_instances,
    _occluder_bbox_ratio,
    _pick_best,
    _union_mask,
    _view_pitch_deg,
)
from facade_grammar.schemas.buildings import Building, Facade
from facade_grammar.schemas.photos import PhotoMetadata


def _photo(pid: str) -> PhotoMetadata:
    return PhotoMetadata(
        photo_id=pid,
        lon=4.9,
        lat=52.37,
        bearing_deg=180.0,
        captured_at=datetime(2024, 6, 1, tzinfo=UTC),
        url=HttpUrl("https://example.invalid/thumb.jpg"),
    )


def _candidate(pid: str, *, rank: int, occluder_ratio: float) -> _Candidate:
    mask = np.zeros((4, 4), dtype=bool)
    return _Candidate(
        photo=_photo(pid),
        rank=rank,
        view_bytes=b"",
        facade_mask=mask,
        facade_bbox=(0, 0, 4, 4),
        occluder_mask=mask,
        facade_score=0.9,
        occluder_ratio=occluder_ratio,
        projected_bbox=None,
        feature_instances_by_class=[],
    )


def test_pick_best_empty_returns_none() -> None:
    assert _pick_best([], max_occluder_ratio=0.4) is None


def test_pick_best_hard_rejects_occluded_then_picks_min_rank() -> None:
    # Above threshold is filtered out; among survivors min rank wins
    # even if another survivor has a lower occluder_ratio.
    heavy = _candidate("heavy", rank=0, occluder_ratio=0.9)
    clean_first = _candidate("first", rank=1, occluder_ratio=0.2)
    clean_cleaner = _candidate("cleaner", rank=2, occluder_ratio=0.1)
    best = _pick_best([heavy, clean_first, clean_cleaner], max_occluder_ratio=0.4)
    assert best is not None and best.photo.photo_id == "first"


def test_pick_best_ties_break_on_rank() -> None:
    early = _candidate("early", rank=0, occluder_ratio=0.2)
    late = _candidate("late", rank=2, occluder_ratio=0.2)
    best = _pick_best([late, early], max_occluder_ratio=0.4)
    assert best is not None and best.photo.photo_id == "early"


def test_pick_best_falls_back_when_all_exceed_threshold() -> None:
    # Rather than drop the facade, pick the best-ranked above-threshold.
    a = _candidate("a", rank=0, occluder_ratio=0.9)
    b = _candidate("b", rank=1, occluder_ratio=0.95)
    best = _pick_best([a, b], max_occluder_ratio=0.4)
    assert best is not None and best.photo.photo_id == "a"


def test_occluder_bbox_ratio_no_overlap_is_zero() -> None:
    # Facade bbox is the top half; occluder sits entirely below it.
    facade_bbox = (0, 0, 4, 2)
    tree = np.zeros((4, 4), dtype=bool)
    tree[2:] = True
    assert _occluder_bbox_ratio(tree, facade_bbox) == 0.0


def test_occluder_bbox_ratio_full_overlap_is_one() -> None:
    facade_bbox = (0, 0, 2, 4)
    tree = np.zeros((4, 4), dtype=bool)
    tree[:, :2] = True
    assert _occluder_bbox_ratio(tree, facade_bbox) == 1.0


def test_occluder_bbox_ratio_half_overlap() -> None:
    # Facade bbox = left half (8 px). Tree covers top-left 4 px inside it.
    facade_bbox = (0, 0, 2, 4)
    tree = np.zeros((4, 4), dtype=bool)
    tree[:2, :2] = True
    assert _occluder_bbox_ratio(tree, facade_bbox) == 0.5


def test_occluder_bbox_ratio_catches_tree_in_facade_hole() -> None:
    # A ring-shaped facade mask has the full frame as its bbox; a tree that
    # fills the hole has zero pixelwise overlap with the ring but the bbox
    # metric sees 4/16 of the bbox covered.
    facade_bbox = (0, 0, 4, 4)
    tree = np.zeros((4, 4), dtype=bool)
    tree[1:3, 1:3] = True
    assert _occluder_bbox_ratio(tree, facade_bbox) == 0.25


def _inst(score: float, box: tuple[float, float, float, float]) -> SamInstance:
    return SamInstance(mask=np.zeros((4, 4), dtype=bool), score=score, box=box)


def test_bbox_of_mask_empty_is_none() -> None:
    assert _bbox_of_mask(np.zeros((4, 4), dtype=bool)) is None


def test_bbox_of_mask_tight_rectangle() -> None:
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:5, 3:8] = True
    assert _bbox_of_mask(mask) == (3, 2, 8, 5)


def test_centre_inside_happy_path() -> None:
    assert _centre_inside((5, 5, 10, 10), (0, 0, 20, 20))


def test_centre_inside_box_overlapping_edge_with_centre_outside_is_rejected() -> None:
    # bbox centre at x=25, outside container max_x=20.
    assert not _centre_inside((20, 5, 30, 10), (0, 0, 20, 20))


def test_filter_instances_drops_low_score_and_outside_centre() -> None:
    container = (0, 0, 100, 100)
    keep = _inst(0.9, (10, 10, 30, 30))
    low_score = _inst(0.1, (10, 10, 30, 30))
    outside = _inst(0.9, (200, 200, 220, 220))
    out = _filter_instances([keep, low_score, outside], container, min_score=0.3)
    assert out == [keep]


def test_union_mask_empty_returns_zeros() -> None:
    out = _union_mask([], shape=(4, 4))
    assert out.shape == (4, 4)
    assert not out.any()


def test_union_mask_pixelwise_or() -> None:
    m1 = np.zeros((4, 4), dtype=bool)
    m1[0, 0] = True
    m2 = np.zeros((4, 4), dtype=bool)
    m2[3, 3] = True
    out = _union_mask(
        [
            SamInstance(mask=m1, score=1.0, box=(0, 0, 1, 1)),
            SamInstance(mask=m2, score=1.0, box=(3, 3, 4, 4)),
        ],
        shape=(4, 4),
    )
    assert out[0, 0] and out[3, 3]
    assert out.sum() == 2


def _pano_at(lon: float, lat: float, altitude_m: float | None = 0.0) -> PhotoMetadata:
    return PhotoMetadata(
        photo_id="p",
        lon=lon,
        lat=lat,
        altitude_m=altitude_m,
        bearing_deg=180.0,
        is_pano=True,
        captured_at=datetime(2024, 6, 1, tzinfo=UTC),
        url=HttpUrl("https://example.invalid/pano.jpg"),
    )


def _facade_north_of_photo(offset_deg: float) -> Facade:
    """Build a canal facade one small ``offset_deg`` north of (4.9, 52.37)."""
    lat = 52.37 + offset_deg
    return Facade(
        building_id="b",
        edge_start=(4.9 - 1e-5, lat),
        edge_end=(4.9 + 1e-5, lat),
        classification="canal",
        normal_deg=180.0,
    )


def _building(height_m: float) -> Building:
    return Building(
        building_id="b",
        footprint=[(4.9, 52.37), (4.9001, 52.37), (4.9001, 52.3701)],
        height_m=height_m,
        ground_altitude_m=0.0,
    )


def test_view_pitch_deg_close_tall_building_requires_positive_pitch() -> None:
    # ~5.5m north of camera at ground, 20m tall → roofline well above half-FoV.
    pitch = _view_pitch_deg(
        _pano_at(4.9, 52.37),
        _facade_north_of_photo(5e-5),
        _building(20.0),
        SamConfig(),
    )
    assert pitch > 0


def test_view_pitch_deg_short_building_clamps_to_zero() -> None:
    # Same geometry as close-tall, but a 1m-tall building — roofline is already
    # inside the default 45deg view without tilting up.
    pitch = _view_pitch_deg(
        _pano_at(4.9, 52.37),
        _facade_north_of_photo(5e-5),
        _building(1.0),
        SamConfig(),
    )
    assert pitch == 0.0


def test_view_pitch_deg_distant_building_clamps_to_zero() -> None:
    # ~500m away, 10m tall → apparent roof elevation < half-FoV, pitch clamps.
    pitch = _view_pitch_deg(
        _pano_at(4.9, 52.37),
        _facade_north_of_photo(0.0045),
        _building(10.0),
        SamConfig(),
    )
    assert pitch == 0.0
