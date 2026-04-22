"""Data ingestion nodes: 3D BAG buildings, OSM streets + waterways, Mapillary photo metadata."""

import polars as pl
from hamilton.function_modifiers import check_output, dataloader, tag
from pydantic import SecretStr

from facade_grammar.clients import bag3d, mapillary, osm
from facade_grammar.config import Bbox, IngestionConfig
from facade_grammar.geo import RD_TO_WGS84, WGS84_TO_RD
from facade_grammar.schemas.buildings import Building
from facade_grammar.schemas.photos import PhotoMetadata
from facade_grammar.schemas.tables import OsmStreetsSchema, OsmWaterwaysSchema


def _expand_bbox(bbox: Bbox, buffer_m: float) -> Bbox:
    """Expand a WGS84 bbox outward by ``buffer_m`` metres (Netherlands-only, via RD-New)."""
    xs, ys = WGS84_TO_RD.transform([bbox.min_lon, bbox.max_lon], [bbox.min_lat, bbox.max_lat])
    new_lons, new_lats = RD_TO_WGS84.transform(
        [xs[0] - buffer_m, xs[1] + buffer_m],
        [ys[0] - buffer_m, ys[1] + buffer_m],
    )
    return Bbox(new_lons[0], new_lats[0], new_lons[1], new_lats[1])


@dataloader()
@tag(stage="ingestion", source="bag3d")
def raw_buildings(area_bbox: Bbox) -> tuple[list[Building], dict[str, str]]:
    data = bag3d.fetch_buildings(area_bbox)
    return data, {"source": "3dbag", "n": str(len(data)), "bbox": str(area_bbox)}


@tag(stage="ingestion")
def buildings_by_id(raw_buildings: list[Building]) -> dict[str, Building]:
    """Id-keyed view of ``raw_buildings`` so per-facade nodes skip the linear scan."""
    return {b.building_id: b for b in raw_buildings}


@check_output(schema=OsmStreetsSchema.to_schema(), importance="fail")
@dataloader()
@tag(stage="ingestion", source="osm")
def raw_streets(area_bbox: Bbox) -> tuple[pl.DataFrame, dict[str, str]]:
    data = osm.fetch_streets(area_bbox)
    return data, {"source": "osm_streets", "n": str(len(data)), "bbox": str(area_bbox)}


@check_output(schema=OsmWaterwaysSchema.to_schema(), importance="fail")
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
    ingestion: IngestionConfig,
) -> tuple[list[PhotoMetadata], dict[str, str]]:
    bbox = _expand_bbox(area_bbox, ingestion.photo_fetch_buffer_m)
    data = mapillary.fetch_photo_metadata(bbox, token=mapillary_token)
    return data, {
        "source": "mapillary",
        "n": str(len(data)),
        "bbox": str(bbox),
        "buffer_m": str(ingestion.photo_fetch_buffer_m),
    }
