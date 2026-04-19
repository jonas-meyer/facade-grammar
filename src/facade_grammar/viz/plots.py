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
from PIL import Image

from facade_grammar.geo import WGS84_GEOD
from facade_grammar.schemas.buildings import (
    Building,
    Facade,
    FacadeFeatures,
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
    ax.set_title(
        f"Canal facades — {total} total, {matched} matched, {total - matched} unmatched"
    )

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
                [x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0],
                color="#00bcd4", linewidth=1.0, linestyle="--",
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
        counts: dict[FeatureClass, int] = dict.fromkeys(_FEATURE_COLORS, 0)
        for cls, path in ff.class_mask_paths.items():
            mask = _load_mask(path)
            if mask.any():
                ax.contour(mask, levels=[0.5], colors=_FEATURE_COLORS[cls], linewidths=1.0)
            counts[cls] = sum(1 for inst in ff.instances if inst.cls == cls)
        for cls in class_counts:
            class_counts[cls].append(counts[cls])
        count_str = " ".join(f"{cls[0]}={counts[cls]}" for cls in _FEATURE_COLORS)
        ax.set_title(f"{ff.building_id[:10]}  {count_str}", fontsize=8)

    means_str = "  ".join(
        f"#{cls}={(sum(vals) / len(vals) if vals else 0.0):.1f}"
        for cls, vals in class_counts.items()
    )
    fig.suptitle(f"{len(facade_features)} facades — mean {means_str}", fontsize=11)
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
        [f.midpoint, WGS84_GEOD.fwd(*f.midpoint, f.normal_deg, distance_m)[:2]]
        for f in facades
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
            ax.add_collection(
                LineCollection(arrow_segments, colors="#e63946", linewidths=1.4)
            )
