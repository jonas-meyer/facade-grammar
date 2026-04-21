"""Snap per-facade SAM detections onto a regularised rectangle lattice.

This is the bridge between the ``FacadeGrammar`` (cluster centres +
split lines) and anything downstream that wants a dense labelled grid per
facade — MDL grammar induction, FaçAID-style parsers, or just human-readable
output.

Algorithm, per facade:

1. Row boundaries come from the grammar: ``[0, gable_y_end?, *floor_split_ys,
   1.0]``. Gable (when present) gets its own topmost row; each detected
   window-row sits in its own row below.
2. Column boundaries are ``[0, *midpoints(column_x_centers), 1.0]`` — cells
   span the midpoint between adjacent column centres, with the outer edges
   at 0 and 1.
3. Each ``FeatureInstance`` is routed to the row x col cell whose bounds
   contain its bbox centre (normalised to the facade bbox, read back from
   the facade mask PNG).
4. Per cell, the highest-scoring detection wins its label. Cells with no
   detection are wall-fills.
"""

import itertools

import numpy as np
from hamilton.function_modifiers import tag

from facade_grammar.schemas.buildings import (
    FacadeFeatures,
    FacadeGrammar,
    FacadeLattice,
    FacadeMask,
    FeatureInstance,
    LatticeCell,
)


@tag(stage="grammar")
def facade_lattices(
    facade_grammars: dict[str, FacadeGrammar],
    facade_features: dict[str, FacadeFeatures],
    facade_masks: dict[str, FacadeMask],
) -> dict[str, FacadeLattice]:
    lattices: dict[str, FacadeLattice] = {}
    for bid, g in facade_grammars.items():
        ff = facade_features.get(bid)
        fm = facade_masks.get(bid)
        if ff is None or fm is None:
            continue
        lattice = _build_lattice(g, ff, fm)
        if lattice is not None:
            lattices[bid] = lattice
    return lattices


def _build_lattice(g: FacadeGrammar, ff: FacadeFeatures, fm: FacadeMask) -> FacadeLattice | None:
    fx0, fy0, fx1, fy1 = fm.facade_bbox_px
    fw, fh = fx1 - fx0, fy1 - fy0
    if fw <= 0 or fh <= 0:
        return None

    row_boundaries = _row_boundaries(g)
    col_boundaries = _col_boundaries(g)
    n_rows = len(row_boundaries) - 1
    n_cols = len(col_boundaries) - 1

    candidates: dict[tuple[int, int], list[FeatureInstance]] = {}
    for inst in ff.instances:
        cx = ((inst.bbox[0] + inst.bbox[2]) / 2 - fx0) / fw
        cy = ((inst.bbox[1] + inst.bbox[3]) / 2 - fy0) / fh
        if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
            continue
        row = _bin_index(cy, row_boundaries)
        col = _bin_index(cx, col_boundaries)
        candidates.setdefault((row, col), []).append(inst)

    cells: list[LatticeCell] = []
    for row, col in itertools.product(range(n_rows), range(n_cols)):
        insts = candidates.get((row, col))
        if not insts:
            cells.append(LatticeCell(row=row, col=col, cls="wall", score=0.0))
            continue
        winner = max(insts, key=lambda i: i.score)
        cells.append(
            LatticeCell(
                row=row,
                col=col,
                cls=winner.cls,
                score=winner.score,
                source_bbox_norm=_norm_bbox(winner.bbox, fx0, fy0, fw, fh),
            )
        )

    return FacadeLattice(
        building_id=g.building_id,
        n_rows=n_rows,
        n_cols=n_cols,
        row_boundaries=row_boundaries,
        col_boundaries=col_boundaries,
        cells=cells,
    )


def _row_boundaries(g: FacadeGrammar) -> list[float]:
    inner = sorted(g.floor_split_ys)
    bounds: list[float] = [0.0]
    if g.gable_y_end is not None and 0.0 < g.gable_y_end < 1.0:
        bounds.append(g.gable_y_end)
    bounds.extend(inner)
    bounds.append(1.0)
    # Deduplicate + enforce strict monotonicity — floor_split_ys may equal the
    # gable boundary on short facades.
    out: list[float] = []
    for b in bounds:
        if not out or b > out[-1] + 1e-9:
            out.append(b)
    return out


def _col_boundaries(g: FacadeGrammar) -> list[float]:
    centers = sorted(g.column_x_centers)
    if not centers:
        return [0.0, 1.0]
    midpoints = [(centers[i] + centers[i + 1]) / 2 for i in range(len(centers) - 1)]
    return [0.0, *midpoints, 1.0]


def _bin_index(value: float, boundaries: list[float]) -> int:
    """Largest ``i`` such that ``boundaries[i] <= value``. Clamped to a
    valid cell index (0 ≤ i ≤ len(boundaries) - 2)."""
    idx = int(np.searchsorted(boundaries, value, side="right")) - 1
    return max(0, min(idx, len(boundaries) - 2))


def _norm_bbox(
    bbox: tuple[int, int, int, int], fx0: int, fy0: int, fw: int, fh: int
) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = bbox
    return (
        float((x0 - fx0) / fw),
        float((y0 - fy0) / fh),
        float((x1 - fx0) / fw),
        float((y1 - fy0) / fh),
    )
