"""Tests for pure helpers in ``facade_grammar.pipeline.vision``."""

from datetime import UTC, datetime

import numpy as np
from pydantic import HttpUrl

from facade_grammar.clients.sam import SamInstance
from facade_grammar.pipeline.vision import (
    _bbox_of_mask,
    _Candidate,
    _centre_inside,
    _filter_instances,
    _occluder_bbox_ratio,
    _pick_best,
    _union_mask,
)
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
        occluder_mask=mask,
        facade_score=0.9,
        occluder_ratio=occluder_ratio,
        projected_bbox=None,
    )


def test_pick_best_empty_returns_none() -> None:
    assert _pick_best([]) is None


def test_pick_best_lowest_occluder_bbox_ratio_wins() -> None:
    a = _candidate("a", rank=0, occluder_ratio=0.4)
    b = _candidate("b", rank=1, occluder_ratio=0.1)
    c = _candidate("c", rank=2, occluder_ratio=0.5)
    best = _pick_best([a, b, c])
    assert best is not None and best.photo.photo_id == "b"


def test_pick_best_ties_break_on_rank() -> None:
    early = _candidate("early", rank=0, occluder_ratio=0.2)
    late = _candidate("late", rank=2, occluder_ratio=0.2)
    best = _pick_best([late, early])
    assert best is not None and best.photo.photo_id == "early"


def test_occluder_bbox_ratio_no_overlap_is_zero() -> None:
    facade = np.zeros((4, 4), dtype=bool)
    facade[:2] = True
    tree = np.zeros((4, 4), dtype=bool)
    tree[2:] = True
    assert _occluder_bbox_ratio(facade, tree) == 0.0


def test_occluder_bbox_ratio_full_overlap_is_one() -> None:
    facade = np.zeros((4, 4), dtype=bool)
    facade[:, :2] = True
    tree = facade.copy()
    assert _occluder_bbox_ratio(facade, tree) == 1.0


def test_occluder_bbox_ratio_half_overlap() -> None:
    facade = np.zeros((4, 4), dtype=bool)
    facade[:, :2] = True  # 8 pixels
    tree = np.zeros((4, 4), dtype=bool)
    tree[:2, :2] = True  # 4 pixels overlap with facade
    assert _occluder_bbox_ratio(facade, tree) == 0.5


def test_occluder_bbox_ratio_empty_facade_is_zero() -> None:
    facade = np.zeros((4, 4), dtype=bool)
    tree = np.ones((4, 4), dtype=bool)
    assert _occluder_bbox_ratio(facade, tree) == 0.0


def test_occluder_bbox_ratio_catches_tree_in_facade_hole() -> None:
    # Tree occludes the centre of a facade: SAM's facade mask is a ring,
    # its tree mask fills the ring. Pixelwise overlap would be 0, but the
    # bbox-based metric correctly sees that the tree is *inside* the facade.
    facade = np.ones((4, 4), dtype=bool)
    facade[1:3, 1:3] = False  # 4-px hole in the middle
    tree = np.zeros((4, 4), dtype=bool)
    tree[1:3, 1:3] = True  # tree fills the hole
    assert _occluder_bbox_ratio(facade, tree) == 0.25  # 4 tree px / 16 bbox px


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
