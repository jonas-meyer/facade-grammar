"""Tests for pure helpers in ``facade_grammar.pipeline.vision``."""

from datetime import UTC, datetime

import numpy as np
from pydantic import HttpUrl

from facade_grammar.pipeline.vision import _Candidate, _occluder_bbox_ratio, _pick_best
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
