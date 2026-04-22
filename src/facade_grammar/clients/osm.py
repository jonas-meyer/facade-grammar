"""OpenStreetMap client (via osmnx) for streets and waterways.

Each function returns a Polars DataFrame with stable row ordering (sorted by
osm_id) so Hamilton's cache fingerprint is deterministic across runs. The
osmnx GeoDataFrame never escapes these functions.
"""

import geopandas as gpd
import osmnx as ox
import polars as pl

from facade_grammar.config import Bbox

ox.settings.use_cache = False

_LINE_TYPES = ("LineString", "MultiLineString")


def fetch_streets(bbox_wgs84: Bbox) -> pl.DataFrame:
    return _fetch_lines(bbox_wgs84, tag="highway")


def fetch_waterways(bbox_wgs84: Bbox) -> pl.DataFrame:
    """Linestring ``waterway=*`` features plus polygon ``natural=water`` surfaces.

    A single Overpass query with both tags returns features that match either,
    and geom-type filtering keeps only line and polygon geometries. Using the
    ``natural=water`` polygon form for most Amsterdam canals gives
    distance-to-quay-edge rather than distance-to-centerline, which is stable
    across canals of different widths.
    """
    gdf = ox.features.features_from_bbox(
        bbox=bbox_wgs84, tags={"waterway": True, "natural": "water"}
    )
    keep_types = (*_LINE_TYPES, "Polygon", "MultiPolygon")
    subset = gdf[gdf.geometry.geom_type.isin(keep_types)]
    if subset.empty:
        return pl.DataFrame(
            schema={
                "osm_id": pl.String,
                "waterway": pl.String,
                "name": pl.String,
                "geometry_wkt": pl.String,
            }
        )
    return pl.DataFrame(
        {
            "osm_id": [str(idx) for idx in subset.index.get_level_values("id")],
            # ``natural=water`` polygons have NaN in the waterway column.
            "waterway": subset["waterway"].fillna("water").astype(str).tolist(),
            "name": _names(subset),
            "geometry_wkt": subset.geometry.to_wkt().tolist(),
        }
    ).sort("osm_id")


def _fetch_lines(bbox_wgs84: Bbox, *, tag: str) -> pl.DataFrame:
    gdf = ox.features.features_from_bbox(bbox=bbox_wgs84, tags={tag: True})
    lines = gdf[gdf.geometry.geom_type.isin(_LINE_TYPES)]
    return pl.DataFrame(
        {
            "osm_id": [str(idx) for idx in lines.index.get_level_values("id")],
            tag: lines[tag].astype(str).tolist(),
            "name": _names(lines),
            "geometry_wkt": lines.geometry.to_wkt().tolist(),
        }
    ).sort("osm_id")


def _names(df: gpd.GeoDataFrame) -> list[str]:
    """osmnx omits the ``name`` column when no feature in the bbox has it."""
    if "name" not in df.columns:
        return [""] * len(df)
    return df["name"].fillna("").astype(str).tolist()
