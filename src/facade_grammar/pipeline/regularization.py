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


_MIN_BOUNDARY_GAP = 0.02


def _row_boundaries(g: FacadeGrammar) -> list[float]:
    """Fine-grained rows: gable + floor splits + each window row's top/bottom."""
    half_h = g.window_h_mean / 2
    points: set[float] = {0.0, 1.0}
    if g.gable_y_end is not None and 0.0 < g.gable_y_end < 1.0:
        points.add(g.gable_y_end)
    points.update(g.floor_split_ys)
    for cy in g.window_row_y_centers:
        points.add(max(0.0, cy - half_h))
        points.add(min(1.0, cy + half_h))
    return _dedupe_close(sorted(points))


def _col_boundaries(g: FacadeGrammar) -> list[float]:
    """Fine-grained columns: window left/right edges around each column centre."""
    centers = sorted(g.column_x_centers)
    if not centers:
        return [0.0, 1.0]
    half_w = g.window_w_mean / 2
    points: set[float] = {0.0, 1.0}
    for cx in centers:
        points.add(max(0.0, cx - half_w))
        points.add(min(1.0, cx + half_w))
    return _dedupe_close(sorted(points))


def _dedupe_close(points: list[float]) -> list[float]:
    """Merge boundaries that fall within ``_MIN_BOUNDARY_GAP`` of each other."""
    if not points:
        return [0.0, 1.0]
    out = [points[0]]
    for p in points[1:]:
        if p > out[-1] + _MIN_BOUNDARY_GAP:
            out.append(p)
    if out[-1] != 1.0:
        out[-1] = 1.0  # force the last boundary to exactly 1 so the final cell closes
    return out


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
