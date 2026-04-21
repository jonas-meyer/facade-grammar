"""Mapillary Graph API client: fetch photo metadata in a WGS84 bbox.

The Graph API enforces an opaque per-request data cap independent of ``limit``;
a dense bbox returns HTTP 500 ("please reduce the amount of data you're asking
for"). We tile the input bbox into an NxN grid and query each tile; tiles
that still 500 get halved recursively.
"""

import itertools
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

import httpx
from pydantic import AliasPath, BaseModel, ConfigDict, Field, HttpUrl, SecretStr

from facade_grammar.config import Bbox
from facade_grammar.schemas.photos import PhotoMetadata

GRAPH_URL = "https://graph.mapillary.com/images"
_IMAGE_URL = "https://graph.mapillary.com/{image_id}"
_FIELDS = (
    "id,computed_geometry,computed_altitude,compass_angle,captured_at,"
    "is_pano,thumb_1024_url,quality_score"
)
_TILE_GRID = 4
_RECURSIVE_SPLIT_MAX_DEPTH = 2


class _MapillaryPhoto(BaseModel):
    """A Mapillary Graph API image record (partial projection of ``_FIELDS``)."""

    model_config = ConfigDict(extra="ignore")

    id: str
    lon: float | None = Field(
        default=None,
        validation_alias=AliasPath("computed_geometry", "coordinates", 0),
    )
    lat: float | None = Field(
        default=None,
        validation_alias=AliasPath("computed_geometry", "coordinates", 1),
    )
    compass_angle: float | None = None
    computed_altitude: float | None = None
    captured_at: int  # milliseconds since the epoch
    is_pano: bool = False
    thumb_1024_url: HttpUrl
    # Mapillary's own photo-quality score, 0-1. Absent (None) on older
    # sequences predating the scoring rollout.
    quality_score: float | None = None

    def to_photo_metadata(self) -> PhotoMetadata | None:
        if self.lon is None or self.lat is None:
            return None
        bearing = (
            None
            if self.compass_angle is None or self.compass_angle < 0
            else float(self.compass_angle)
        )
        return PhotoMetadata(
            photo_id=self.id,
            lon=self.lon,
            lat=self.lat,
            bearing_deg=bearing,
            altitude_m=self.computed_altitude,
            captured_at=datetime.fromtimestamp(self.captured_at / 1000, tz=UTC),
            is_pano=self.is_pano,
            url=self.thumb_1024_url,
            quality_score=self.quality_score,
        )


def fetch_image_bytes(url: HttpUrl | str, *, timeout_s: float = 60.0) -> bytes:
    """Download a Mapillary thumb by its signed URL; no OAuth header needed."""
    with httpx.Client(timeout=timeout_s) as client:
        resp = client.get(str(url))
        resp.raise_for_status()
        return resp.content


def fetch_thumb_url(
    photo_id: str,
    *,
    field: str,
    token: SecretStr,
    timeout_s: float = 30.0,
) -> str:
    """Fetch a larger thumb URL for one image (e.g. ``thumb_2048_url``).

    The bbox query can't include these heavy fields without tipping Mapillary's
    opaque data cap, so we request them per-image at SAM time only for the
    photos we actually segment.
    """
    with httpx.Client(timeout=timeout_s) as client:
        resp = client.get(
            _IMAGE_URL.format(image_id=photo_id),
            params={"fields": field},
            headers={"Authorization": f"OAuth {token.get_secret_value()}"},
        )
        resp.raise_for_status()
        return resp.json()[field]


def fetch_photo_metadata(
    bbox_wgs84: Bbox,
    *,
    token: SecretStr,
) -> list[PhotoMetadata]:
    headers = {"Authorization": f"OAuth {token.get_secret_value()}"}
    tiles = _tile_bbox(bbox_wgs84, grid=_TILE_GRID)

    seen_ids: set[str] = set()
    photos: list[PhotoMetadata] = []
    with httpx.Client(timeout=60.0) as client:
        for tile in tiles:
            for item in _stream_tile(client, headers, tile, depth=0):
                raw_id = item.get("id")
                if not isinstance(raw_id, str) or raw_id in seen_ids:
                    continue
                photo = _MapillaryPhoto.model_validate(item).to_photo_metadata()
                if photo is None:
                    continue
                seen_ids.add(photo.photo_id)
                photos.append(photo)
    return photos


def _tile_bbox(bbox: Bbox, *, grid: int) -> Iterator[Bbox]:
    min_lon, min_lat, max_lon, max_lat = bbox
    lon_step = (max_lon - min_lon) / grid
    lat_step = (max_lat - min_lat) / grid
    for ix, iy in itertools.product(range(grid), range(grid)):
        yield Bbox(
            min_lon + ix * lon_step,
            min_lat + iy * lat_step,
            min_lon + (ix + 1) * lon_step,
            min_lat + (iy + 1) * lat_step,
        )


def _stream_tile(
    client: httpx.Client,
    headers: dict[str, str],
    bbox: Bbox,
    *,
    depth: int,
) -> Iterator[dict[str, Any]]:
    url: str | None = GRAPH_URL
    params: dict[str, str] | None = {
        "fields": _FIELDS,
        "bbox": ",".join(str(v) for v in bbox),
    }
    while url is not None:
        resp = client.get(url, params=params, headers=headers)
        if resp.status_code == 500 and depth < _RECURSIVE_SPLIT_MAX_DEPTH:
            for sub in _halve(bbox):
                yield from _stream_tile(client, headers, sub, depth=depth + 1)
            return
        resp.raise_for_status()
        payload = resp.json()
        yield from payload.get("data", [])
        url = payload.get("paging", {}).get("next")
        params = None  # next-url already has its query string baked in


def _halve(bbox: Bbox) -> Iterator[Bbox]:
    min_lon, min_lat, max_lon, max_lat = bbox
    mid_lon = (min_lon + max_lon) / 2
    mid_lat = (min_lat + max_lat) / 2
    yield Bbox(min_lon, min_lat, mid_lon, mid_lat)
    yield Bbox(mid_lon, min_lat, max_lon, mid_lat)
    yield Bbox(min_lon, mid_lat, mid_lon, max_lat)
    yield Bbox(mid_lon, mid_lat, max_lon, max_lat)
