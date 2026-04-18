"""Shared WGS84 <-> EPSG:28992 (RD New) transformers and geometry helpers."""

from collections.abc import Iterable

import shapely
import shapely.ops
from pyproj import Transformer
from shapely.geometry.base import BaseGeometry

WGS84_TO_RD = Transformer.from_crs(4326, 28992, always_xy=True)
RD_TO_WGS84 = Transformer.from_crs(28992, 4326, always_xy=True)


def build_rd_geoms(wkts: Iterable[str]) -> list[BaseGeometry]:
    """Parse WGS84-WKT geometries and reproject each to EPSG:28992."""
    return [shapely.ops.transform(WGS84_TO_RD.transform, shapely.from_wkt(w)) for w in wkts]
