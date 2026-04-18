"""Data ingestion nodes: 3D BAG buildings, OSM streets + waterways, Mapillary photo metadata."""

import polars as pl
from hamilton.function_modifiers import dataloader, tag
from pydantic import SecretStr

from facade_grammar.clients import bag3d, mapillary, osm
from facade_grammar.config import Bbox
from facade_grammar.schemas.buildings import Building
from facade_grammar.schemas.photos import PhotoMetadata


@dataloader()
@tag(stage="ingestion", source="bag3d")
def raw_buildings(area_bbox: Bbox) -> tuple[list[Building], dict[str, str]]:
    data = bag3d.fetch_buildings(area_bbox)
    return data, {"source": "3dbag", "n": str(len(data)), "bbox": str(area_bbox)}


@dataloader()
@tag(stage="ingestion", source="osm")
def raw_streets(area_bbox: Bbox) -> tuple[pl.DataFrame, dict[str, str]]:
    data = osm.fetch_streets(area_bbox)
    return data, {"source": "osm_streets", "n": str(len(data)), "bbox": str(area_bbox)}


@dataloader()
@tag(stage="ingestion", source="osm")
def raw_waterways(area_bbox: Bbox) -> tuple[pl.DataFrame, dict[str, str]]:
    data = osm.fetch_waterways(area_bbox)
    return data, {"source": "osm_waterways", "n": str(len(data)), "bbox": str(area_bbox)}


@dataloader()
@tag(stage="ingestion", source="mapillary")
def raw_photo_metadata(
    area_bbox: Bbox,
    mapillary_token: SecretStr,
) -> tuple[list[PhotoMetadata], dict[str, str]]:
    data = mapillary.fetch_photo_metadata(area_bbox, token=mapillary_token)
    return data, {
        "source": "mapillary",
        "n": str(len(data)),
        "bbox": str(area_bbox),
    }
