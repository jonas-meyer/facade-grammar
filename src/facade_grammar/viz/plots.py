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
from pyproj import Geod

from facade_grammar.schemas.buildings import Building, Facade
from facade_grammar.schemas.photos import PhotoMetadata

_GEOD = Geod(ellps="WGS84")


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
        [f.midpoint, _GEOD.fwd(*f.midpoint, f.normal_deg, distance_m)[:2]]
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
            [(p.lon, p.lat), _GEOD.fwd(p.lon, p.lat, p.bearing_deg, arrow_m)[:2]]
            for _, p in matched
            if p.bearing_deg is not None
        ]
        if arrow_segments:
            ax.add_collection(
                LineCollection(arrow_segments, colors="#e63946", linewidths=1.4)
            )
