"""Pandera schemas for polars DataFrames flowing through the pipeline.

Ingestion nodes return Polars DataFrames (via ``clients.osm``); downstream
spatial/viz code reads specific columns. These schemas pin the wire contract
between ingestion and consumers so an upstream format drift fails fast with a
clear error instead of surfacing deep in spatial analysis.
"""

import pandera.polars as pa


class OsmStreetsSchema(pa.DataFrameModel):
    """OSM highway linestrings in WKT form, returned by ``osm.fetch_streets``."""

    osm_id: str
    highway: str
    name: str
    geometry_wkt: str


class OsmWaterwaysSchema(pa.DataFrameModel):
    """OSM waterway linestrings + ``natural=water`` polygons in WKT form."""

    osm_id: str
    waterway: str
    name: str
    geometry_wkt: str
