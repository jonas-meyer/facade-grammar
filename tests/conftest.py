"""Shared pytest fixtures."""

from collections.abc import Callable
from typing import Literal, cast

import pytest

from facade_grammar.schemas.buildings import FacadeLattice, FeatureClass, LatticeCell

_CellCls = FeatureClass | Literal["wall"]


@pytest.fixture
def make_lattice() -> Callable[[str, list[list[str]]], FacadeLattice]:
    def _make(bid: str, grid: list[list[str]]) -> FacadeLattice:
        n_rows = len(grid)
        n_cols = len(grid[0])
        return FacadeLattice(
            building_id=bid,
            n_rows=n_rows,
            n_cols=n_cols,
            row_boundaries=[r / n_rows for r in range(n_rows + 1)],
            col_boundaries=[c / n_cols for c in range(n_cols + 1)],
            cells=[
                LatticeCell(row=r, col=c, cls=cast(_CellCls, cls), score=1.0)
                for r, row in enumerate(grid)
                for c, cls in enumerate(row)
            ],
        )

    return _make
