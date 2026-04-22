"""Tests for ``facade_grammar.pipeline.regularization._build_lattice``."""

from pathlib import Path
from typing import Any

from facade_grammar.pipeline.regularization import _build_lattice
from facade_grammar.schemas.buildings import (
    FacadeFeatures,
    FacadeGrammar,
    FacadeMask,
    FeatureClass,
    FeatureInstance,
)


def _grammar(**overrides: Any) -> FacadeGrammar:
    base: dict[str, Any] = {
        "building_id": "b1",
        "photo_id": "p",
        "n_floors": 2,
        "n_columns": 2,
        "window_row_y_centers": [0.25, 0.75],
        "floor_split_ys": [0.5],
        "column_x_centers": [0.25, 0.75],
        "window_w_mean": 0.1,
        "window_w_std": 0.0,
        "window_h_mean": 0.1,
        "window_h_std": 0.0,
        "n_windows_used": 4,
        "n_doors_used": 0,
        "facade_width_m": 10.0,
        "facade_height_m": 12.0,
        "window_w_mean_m": 1.0,
        "window_h_mean_m": 1.2,
        "floor_height_m": 6.0,
    }
    base.update(overrides)
    return FacadeGrammar(**base)


def _mask(facade_bbox_px: tuple[int, int, int, int] = (0, 0, 100, 100)) -> FacadeMask:
    return FacadeMask(
        building_id="b1",
        photo_id="p",
        view_path=Path("/tmp/v.jpg"),
        mask_path=Path("/tmp/m.png"),
        occluder_mask_path=Path("/tmp/o.png"),
        facade_score=0.9,
        occluder_ratio=0.0,
        projected_bbox=None,
        facade_bbox_px=facade_bbox_px,
    )


def _features(*instances: FeatureInstance) -> FacadeFeatures:
    return FacadeFeatures(
        building_id="b1",
        photo_id="p",
        view_path=Path("/tmp/v.jpg"),
        class_mask_paths={},
        instances=list(instances),
    )


def _window(x: int, y: int, score: float = 0.9, cls: FeatureClass = "window") -> FeatureInstance:
    return FeatureInstance(cls=cls, score=score, bbox=(x - 5, y - 5, x + 5, y + 5))


def test_build_lattice_places_each_window_in_exactly_one_cell() -> None:
    lattice = _build_lattice(
        _grammar(),
        _features(_window(25, 25), _window(75, 25), _window(25, 75), _window(75, 75)),
        _mask(),
    )
    assert lattice is not None
    windows = [c for c in lattice.cells if c.cls == "window"]
    walls = [c for c in lattice.cells if c.cls == "wall"]
    # Four input windows → four window cells; the rest are wall (finer lattice).
    assert len(windows) == 4
    assert len(walls) == lattice.n_rows * lattice.n_cols - 4


def test_build_lattice_empty_cells_default_to_wall() -> None:
    lattice = _build_lattice(_grammar(), _features(_window(25, 25)), _mask())
    assert lattice is not None
    by_class: dict[str, int] = {}
    for cell in lattice.cells:
        by_class[cell.cls] = by_class.get(cell.cls, 0) + 1
    # One feature → one non-wall cell; everything else is wall.
    assert by_class.get("window", 0) == 1
    assert by_class.get("wall", 0) == lattice.n_rows * lattice.n_cols - 1


def test_build_lattice_picks_highest_scoring_detection_per_cell() -> None:
    lattice = _build_lattice(
        _grammar(),
        _features(_window(25, 25, score=0.4), _window(27, 27, score=0.9, cls="door")),
        _mask(),
    )
    assert lattice is not None
    # Both bbox centres (25,25 and 27,27) normalise to ~0.25 — land in the
    # same cell with the new finer boundary scheme. Winner is the higher score.
    winners = [c for c in lattice.cells if c.cls != "wall"]
    assert len(winners) == 1
    assert winners[0].cls == "door"
    assert winners[0].score == 0.9


def test_build_lattice_drops_detections_outside_facade_bbox() -> None:
    # bbox centre at (150, 150) is outside the 100x100 facade bbox.
    outside = FeatureInstance(cls="window", score=0.9, bbox=(145, 145, 155, 155))
    lattice = _build_lattice(_grammar(), _features(outside), _mask())
    assert lattice is not None
    assert all(c.cls == "wall" for c in lattice.cells)


def test_build_lattice_returns_none_for_zero_area_facade_bbox() -> None:
    assert _build_lattice(_grammar(), _features(), _mask(facade_bbox_px=(50, 50, 50, 50))) is None
