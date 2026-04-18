"""3D BAG WFS client: fetch Dutch buildings in a WGS84 bounding box."""

from collections.abc import Iterator
from typing import Any

import httpx
import shapely
import shapely.geometry
from pydantic import AliasPath, BaseModel, ConfigDict, Field
from pyproj import Transformer

from facade_grammar.config import Bbox
from facade_grammar.schemas.buildings import Building

WFS_URL = "https://data.3dbag.nl/api/BAG3D/wfs"
_TYPE_NAME = "BAG3D:lod12"
_PAGE_SIZE = 1000

_WGS84_TO_RD = Transformer.from_crs(4326, 28992, always_xy=True)
_RD_TO_WGS84 = Transformer.from_crs(28992, 4326, always_xy=True)


class _BagBuilding(BaseModel):
    """A 3D BAG WFS feature (``BAG3D:lod12`` layer)."""

    model_config = ConfigDict(extra="ignore")

    geometry: dict[str, Any]
    identificatie: str = Field(validation_alias=AliasPath("properties", "identificatie"))
    b3_h_70p: float = Field(validation_alias=AliasPath("properties", "b3_h_70p"))
    b3_h_maaiveld: float = Field(validation_alias=AliasPath("properties", "b3_h_maaiveld"))

    def to_building(self) -> Building:
        polygon = shapely.get_parts(shapely.geometry.shape(self.geometry))[0]
        ring = list(polygon.exterior.coords)
        lons, lats = _RD_TO_WGS84.transform([p[0] for p in ring], [p[1] for p in ring])
        return Building(
            building_id=self.identificatie,
            footprint=list(zip(lons, lats, strict=True)),
            height_m=self.b3_h_70p - self.b3_h_maaiveld,
        )


def fetch_buildings(bbox_wgs84: Bbox) -> list[Building]:
    min_x, min_y, max_x, max_y = _WGS84_TO_RD.transform_bounds(*bbox_wgs84)
    rd_bbox = f"{min_x},{min_y},{max_x},{max_y},EPSG:28992"

    with httpx.Client(timeout=60.0) as client:
        return [
            _BagBuilding.model_validate(feat).to_building()
            for feat in _paginate(client, rd_bbox, page_size=_PAGE_SIZE)
        ]


def _paginate(client: httpx.Client, rd_bbox: str, *, page_size: int) -> Iterator[dict[str, Any]]:
    start_index = 0
    while True:
        resp = client.get(
            WFS_URL,
            params={
                "service": "WFS",
                "version": "2.0.0",
                "request": "GetFeature",
                "typeNames": _TYPE_NAME,
                "bbox": rd_bbox,
                "count": str(page_size),
                "startIndex": str(start_index),
                "outputFormat": "application/json",
            },
        )
        resp.raise_for_status()
        features = resp.json().get("features", [])
        yield from features
        if len(features) < page_size:
            return
        start_index += len(features)
