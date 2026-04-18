"""Matplotlib plotting helpers for debug-stage Hamilton nodes.

``matplotlib.use("Agg")`` is called before ``pyplot`` is imported so this
module works in headless environments.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import polars as pl
import shapely
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection, PolyCollection

from facade_grammar.schemas.buildings import Building
from facade_grammar.schemas.photos import PhotoMetadata


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
    segments = [
        list(part.coords)
        for wkt in df[column]
        for part in shapely.get_parts(shapely.from_wkt(wkt))
    ]
    if not segments:
        return
    ax.add_collection(
        LineCollection(segments, colors=color, linewidths=lw, label=label)
    )


def _draw_photos(ax: Axes, photos: list[PhotoMetadata]) -> None:
    if not photos:
        return
    lons = [p.lon for p in photos]
    lats = [p.lat for p in photos]
    ax.scatter(lons, lats, s=6, c="#e63946", label="Mapillary photos", zorder=5)
