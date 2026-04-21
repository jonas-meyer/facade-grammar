"""Matplotlib plotting helpers for debug-stage Hamilton nodes.

``matplotlib.use("Agg")`` is called before ``pyplot`` is imported so this
module works in headless environments.
"""

import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import shapely
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.patches import Rectangle
from PIL import Image

from facade_grammar.geo import WGS84_GEOD
from facade_grammar.schemas.buildings import (
    Building,
    Facade,
    FacadeFeatures,
    FacadeGrammar,
    FacadeLattice,
    FacadeMask,
    FeatureClass,
)
from facade_grammar.schemas.photos import PhotoMetadata

_CONTACT_SHEET_ROWS = 4
_CONTACT_SHEET_COLS = 3
_CONTACT_SHEET_SEED = 42
_FEATURE_COLORS: dict[FeatureClass, str] = {
    "window": "#1f77b4",
    "door": "#ff7f0e",
    "gable": "#9467bd",
    "floor": "#f1c40f",
    "chimney": "#8c564b",
    "balcony": "#e377c2",
    "hoist_beam": "#17becf",
}


def plot_area_map(
    *,
    buildings: list[Building],
    streets: pl.DataFrame,
    waterways: pl.DataFrame,
    photos: list[PhotoMetadata],
    out_path: Path,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect("equal")

    _draw_buildings(ax, buildings)
    _draw_lines(ax, streets, column="geometry_wkt", color="#6a6a6a", lw=0.8, label="Streets (OSM)")
    _draw_lines(ax, waterways, column="geometry_wkt", color="#2e86ab", lw=1.4, label="Canals (OSM)")
    _draw_photos(ax, photos)

    ax.autoscale_view()
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(loc="upper left", fontsize=8, frameon=True)
    ax.set_title(f"Test area — {len(buildings)} buildings, {len(photos)} photos")

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_canal_selection(
    *,
    buildings: list[Building],
    streets: pl.DataFrame,
    waterways: pl.DataFrame,
    canal_facades: list[Facade],
    best_photo_per_facade: dict[str, PhotoMetadata],
    out_path: Path,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect("equal")

    _draw_buildings(ax, buildings)
    _draw_lines(ax, streets, column="geometry_wkt", color="#6a6a6a", lw=0.8, label="Streets (OSM)")
    _draw_lines(ax, waterways, column="geometry_wkt", color="#2e86ab", lw=1.4, label="Canals (OSM)")
    _draw_canal_facades(ax, canal_facades)
    _draw_normal_ticks(ax, canal_facades, distance_m=5.0)
    _draw_selections(ax, canal_facades, best_photo_per_facade, arrow_m=8.0)

    ax.autoscale_view()
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(loc="upper left", fontsize=8, frameon=True)
    matched = len(best_photo_per_facade)
    total = len(canal_facades)
    ax.set_title(f"Canal facades — {total} total, {matched} matched, {total - matched} unmatched")

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_facade_mask_contact_sheet(
    *,
    facade_masks: dict[str, FacadeMask],
    out_path: Path,
) -> Path:
    """Grid of selected facades with SAM masks + projected bbox for eyeball QA."""
    fig, cells, sample = _init_contact_sheet(list(facade_masks.values()), out_path)

    scores: list[float] = []
    ratios: list[float] = []
    for ax, fm in zip(cells, sample, strict=False):
        image = Image.open(fm.view_path).convert("RGB")
        ax.imshow(np.asarray(image))
        facade = _load_mask(fm.mask_path)
        occluder = _load_mask(fm.occluder_mask_path)
        if facade.any():
            ax.contour(facade, levels=[0.5], colors="#2ca02c", linewidths=1.4)
        if occluder.any():
            ax.contour(occluder, levels=[0.5], colors="#e63946", linewidths=0.9)
        if fm.projected_bbox is not None:
            x0, y0, x1, y1 = fm.projected_bbox
            ax.plot(
                [x0, x1, x1, x0, x0],
                [y0, y0, y1, y1, y0],
                color="#00bcd4",
                linewidth=1.0,
                linestyle="--",
            )
        ax.set_title(
            f"{fm.building_id[:10]}  s={fm.facade_score:.2f}  occ={fm.occluder_ratio:.0%}",
            fontsize=8,
        )
        scores.append(fm.facade_score)
        ratios.append(fm.occluder_ratio)

    mean_s = sum(scores) / len(scores) if scores else 0.0
    mean_r = sum(ratios) / len(ratios) if ratios else 0.0
    fig.suptitle(
        f"{len(facade_masks)} facades masked — mean score {mean_s:.2f}, "
        f"mean occluder coverage {mean_r:.0%}",
        fontsize=11,
    )
    return _save_contact_sheet(fig, out_path)


def plot_facade_features_contact_sheet(
    *,
    facade_features: dict[str, FacadeFeatures],
    out_path: Path,
) -> Path:
    """Grid of facades with per-class feature outlines (window/door/gable/floor)."""
    fig, cells, sample = _init_contact_sheet(list(facade_features.values()), out_path)

    class_counts: dict[FeatureClass, list[int]] = {cls: [] for cls in _FEATURE_COLORS}
    for ax, ff in zip(cells, sample, strict=False):
        image = Image.open(ff.view_path).convert("RGB")
        ax.imshow(np.asarray(image))
        counts: dict[FeatureClass, int] = {}
        for cls in _FEATURE_COLORS:
            path = ff.class_mask_paths.get(cls)
            if path is not None and (mask := _load_mask(path)).any():
                ax.contour(mask, levels=[0.5], colors=_FEATURE_COLORS[cls], linewidths=1.0)
            counts[cls] = sum(1 for inst in ff.instances if inst.cls == cls)
            class_counts[cls].append(counts[cls])
        count_str = " ".join(f"{cls[0]}={counts[cls]}" for cls in _FEATURE_COLORS)
        ax.set_title(f"{ff.building_id[:10]}  {count_str}", fontsize=8)

    means_str = "  ".join(
        f"#{cls}={(sum(vals) / len(vals) if vals else 0.0):.1f}"
        for cls, vals in class_counts.items()
    )
    fig.suptitle(f"{len(facade_features)} facades — mean {means_str}", fontsize=11)
    return _save_contact_sheet(fig, out_path)


def plot_facade_grammar_contact_sheet(
    *,
    facade_grammars: dict[str, FacadeGrammar],
    facade_masks: dict[str, FacadeMask],
    out_path: Path,
) -> Path:
    """Grid of facades with fitted floor/column/gable/door overlaid for eyeball QA."""
    pairs = [(g, facade_masks[bid]) for bid, g in facade_grammars.items() if bid in facade_masks]
    fig, cells, sample = _init_contact_sheet(pairs, out_path)

    for ax, (g, fm) in zip(cells, sample, strict=False):
        image = Image.open(fm.view_path).convert("RGB")
        ax.imshow(np.asarray(image))
        fx0, fy0, fx1, fy1 = fm.facade_bbox_px
        fw, fh = fx1 - fx0, fy1 - fy0

        if g.gable_y_end is not None:
            ax.add_patch(
                Rectangle(
                    (fx0, fy0),
                    fw,
                    g.gable_y_end * fh,
                    color="#f1c40f",
                    alpha=0.25,
                    linewidth=0,
                )
            )
        ax.plot(
            [fx0, fx1, fx1, fx0, fx0],
            [fy0, fy0, fy1, fy1, fy0],
            color="#2ca02c",
            linewidth=1.0,
        )
        for y in g.floor_split_ys:
            y_abs = fy0 + y * fh
            ax.plot([fx0, fx1], [y_abs, y_abs], color="#1f77b4", linewidth=1.2, alpha=0.9)
        for y in g.window_row_y_centers:
            y_abs = fy0 + y * fh
            ax.plot(
                [fx0, fx1],
                [y_abs, y_abs],
                color="#1f77b4",
                linewidth=0.5,
                alpha=0.35,
                linestyle="--",
            )
        for x in g.column_x_centers:
            x_abs = fx0 + x * fw
            ax.plot([x_abs, x_abs], [fy0, fy1], color="#9467bd", linewidth=0.8, alpha=0.75)
        if g.door_x_center is not None and g.door_floor_idx is not None:
            sorted_rows = sorted(g.window_row_y_centers)
            if 0 <= g.door_floor_idx < len(sorted_rows):
                dy = fy0 + sorted_rows[g.door_floor_idx] * fh
                dx = fx0 + g.door_x_center * fw
                ax.scatter(
                    [dx],
                    [dy],
                    s=80,
                    c="#ff7f0e",
                    marker="v",
                    edgecolors="black",
                    linewidths=1.2,
                    zorder=5,
                )

        ax.set_title(
            f"{g.building_id[:10]}  f={g.n_floors} c={g.n_columns} "
            f"w={g.n_windows_used} d={g.n_doors_used}",
            fontsize=8,
        )

    total = len(facade_grammars)
    mean_floors = sum(v.n_floors for v in facade_grammars.values()) / max(1, total)
    mean_cols = sum(v.n_columns for v in facade_grammars.values()) / max(1, total)
    fig.suptitle(
        f"{total} grammars — mean floors={mean_floors:.1f}  columns={mean_cols:.1f}",
        fontsize=11,
    )
    return _save_contact_sheet(fig, out_path)


def plot_facade_lattice_contact_sheet(
    *,
    facade_lattices: dict[str, FacadeLattice],
    facade_masks: dict[str, FacadeMask],
    out_path: Path,
) -> Path:
    """Per-facade overlay of the regularised lattice — grid lines + cell
    fills coloured by dominant class."""
    pairs = [
        (lat, facade_masks[bid]) for bid, lat in facade_lattices.items() if bid in facade_masks
    ]
    fig, cells_ax, sample = _init_contact_sheet(pairs, out_path)

    total_cells = 0
    filled_cells = 0
    for ax, (lat, fm) in zip(cells_ax, sample, strict=False):
        image = Image.open(fm.view_path).convert("RGB")
        ax.imshow(np.asarray(image))
        fx0, fy0, fx1, fy1 = fm.facade_bbox_px
        fw, fh = fx1 - fx0, fy1 - fy0

        for cell in lat.cells:
            total_cells += 1
            if cell.cls == "wall":
                continue
            filled_cells += 1
            y0_rel = lat.row_boundaries[cell.row]
            y1_rel = lat.row_boundaries[cell.row + 1]
            x0_rel = lat.col_boundaries[cell.col]
            x1_rel = lat.col_boundaries[cell.col + 1]
            color = _FEATURE_COLORS.get(cell.cls, "#999999")
            ax.add_patch(
                Rectangle(
                    (fx0 + x0_rel * fw, fy0 + y0_rel * fh),
                    (x1_rel - x0_rel) * fw,
                    (y1_rel - y0_rel) * fh,
                    color=color,
                    alpha=0.35,
                    linewidth=0,
                )
            )

        for y_rel in lat.row_boundaries:
            y_abs = fy0 + y_rel * fh
            ax.plot([fx0, fx1], [y_abs, y_abs], color="white", linewidth=0.7, alpha=0.9)
        for x_rel in lat.col_boundaries:
            x_abs = fx0 + x_rel * fw
            ax.plot([x_abs, x_abs], [fy0, fy1], color="white", linewidth=0.7, alpha=0.9)

        non_wall = sum(1 for c in lat.cells if c.cls != "wall")
        ax.set_title(
            f"{lat.building_id[:10]}  {lat.n_rows}x{lat.n_cols}  "
            f"filled={non_wall}/{lat.n_rows * lat.n_cols}",
            fontsize=8,
        )

    fill_rate = 100 * filled_cells / max(total_cells, 1)
    fig.suptitle(
        f"{len(facade_lattices)} lattices — {fill_rate:.0f}% cells filled (rest wall)",
        fontsize=11,
    )
    return _save_contact_sheet(fig, out_path)


def _init_contact_sheet[T](
    items: list[T], out_path: Path
) -> tuple[plt.Figure, list[Axes], list[T]]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_cells = _CONTACT_SHEET_ROWS * _CONTACT_SHEET_COLS
    rng = random.Random(_CONTACT_SHEET_SEED)
    sample = rng.sample(items, min(n_cells, len(items)))
    fig, axes = plt.subplots(_CONTACT_SHEET_ROWS, _CONTACT_SHEET_COLS, figsize=(12, 14))
    for ax in axes.flat:
        ax.axis("off")
    return fig, list(axes.flat), sample


def _save_contact_sheet(fig: plt.Figure, out_path: Path) -> Path:
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _load_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path), dtype=bool)


def _draw_buildings(ax: Axes, buildings: list[Building]) -> None:
    verts = [b.footprint for b in buildings if len(b.footprint) >= 3]
    if not verts:
        return
    ax.add_collection(
        PolyCollection(
            verts,
            facecolor="#d0d0d0",
            edgecolor="#606060",
            linewidth=0.4,
            label="Buildings (3D BAG)",
        )
    )


def _draw_lines(
    ax: Axes, df: pl.DataFrame, *, column: str, color: str, lw: float, label: str
) -> None:
    segments: list[list[tuple[float, float]]] = []
    for wkt in df[column]:
        geom = shapely.from_wkt(wkt)
        # Polygons come from natural=water; draw their boundary as lines.
        if geom.geom_type in ("Polygon", "MultiPolygon"):
            geom = geom.boundary
        for part in shapely.get_parts(geom):
            segments.append(list(part.coords))
    if not segments:
        return
    ax.add_collection(LineCollection(segments, colors=color, linewidths=lw, label=label))


def _draw_photos(ax: Axes, photos: list[PhotoMetadata]) -> None:
    if not photos:
        return
    lons = [p.lon for p in photos]
    lats = [p.lat for p in photos]
    ax.scatter(lons, lats, s=6, c="#e63946", label="Mapillary photos", zorder=5)


def _draw_canal_facades(ax: Axes, facades: list[Facade]) -> None:
    segments = [[f.edge_start, f.edge_end] for f in facades]
    if not segments:
        return
    ax.add_collection(
        LineCollection(segments, colors="#f4a261", linewidths=2.5, label="Canal facades")
    )


def _draw_normal_ticks(ax: Axes, facades: list[Facade], *, distance_m: float) -> None:
    segments = [
        [f.midpoint, WGS84_GEOD.fwd(*f.midpoint, f.normal_deg, distance_m)[:2]] for f in facades
    ]
    if segments:
        ax.add_collection(LineCollection(segments, colors="#f4a261", linewidths=0.8))


def _draw_selections(
    ax: Axes,
    facades: list[Facade],
    best_photo_per_facade: dict[str, PhotoMetadata],
    *,
    arrow_m: float,
) -> None:
    matched: list[tuple[Facade, PhotoMetadata]] = [
        (f, best_photo_per_facade[f.building_id])
        for f in facades
        if f.building_id in best_photo_per_facade
    ]
    unmatched = [f for f in facades if f.building_id not in best_photo_per_facade]

    if unmatched:
        ax.scatter(
            [f.midpoint[0] for f in unmatched],
            [f.midpoint[1] for f in unmatched],
            s=30,
            facecolors="none",
            edgecolors="#f4a261",
            linewidths=1.3,
            label="Unmatched canal facades",
            zorder=6,
        )

    if matched:
        ax.scatter(
            [p.lon for _, p in matched],
            [p.lat for _, p in matched],
            s=12,
            c="#e63946",
            label="Selected photos",
            zorder=6,
        )
        ax.add_collection(
            LineCollection(
                [[(p.lon, p.lat), f.midpoint] for f, p in matched],
                colors="#e63946",
                linewidths=0.5,
                linestyles="dashed",
            )
        )
        arrow_segments = [
            [(p.lon, p.lat), WGS84_GEOD.fwd(p.lon, p.lat, p.bearing_deg, arrow_m)[:2]]
            for _, p in matched
            if p.bearing_deg is not None
        ]
        if arrow_segments:
            ax.add_collection(LineCollection(arrow_segments, colors="#e63946", linewidths=1.4))
