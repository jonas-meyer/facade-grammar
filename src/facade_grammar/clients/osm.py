"""OpenStreetMap client (via osmnx) for streets and waterways.

Each function returns a Polars DataFrame with stable row ordering (sorted by
osm_id) so Hamilton's cache fingerprint is deterministic across runs. The
osmnx GeoDataFrame never escapes these functions.
"""

import osmnx as ox
import polars as pl

from facade_grammar.config import Bbox

ox.settings.use_cache = False

_LINE_TYPES = ("LineString", "MultiLineString")


def fetch_streets(bbox_wgs84: Bbox) -> pl.DataFrame:
    return _fetch_lines(bbox_wgs84, tag="highway")


def fetch_waterways(bbox_wgs84: Bbox) -> pl.DataFrame:
    return _fetch_lines(bbox_wgs84, tag="waterway")


def _fetch_lines(bbox_wgs84: Bbox, *, tag: str) -> pl.DataFrame:
    gdf = ox.features.features_from_bbox(bbox=bbox_wgs84, tags={tag: True})
    keep_cols = [c for c in (tag, "name") if c in gdf.columns] + ["geometry"]
    lines = gdf.loc[gdf.geometry.geom_type.isin(_LINE_TYPES), keep_cols]
    names = (
        ["" if n is None else str(n) for n in lines["name"].tolist()]
        if "name" in lines.columns
        else [""] * len(lines)
    )
    return pl.DataFrame(
        {
            "osm_id": [str(idx) for idx in lines.index.get_level_values("id")],
            tag: lines[tag].astype(str).tolist(),
            "name": names,
            "geometry_wkt": lines.geometry.to_wkt().tolist(),
        }
    ).sort("osm_id")
