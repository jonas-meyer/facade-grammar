"""Mapillary photo metadata records."""

from datetime import datetime

from pydantic import BaseModel, Field, HttpUrl


class PhotoMetadata(BaseModel):
    """Single Mapillary photo with camera pose.

    ``bearing_deg`` is ``None`` when Mapillary has no SfM-refined heading for
    the image (its ``compass_angle`` comes back as ``-1``).
    """

    photo_id: str
    lon: float
    lat: float
    bearing_deg: float | None = Field(default=None, ge=0, lt=360)
    captured_at: datetime
    url: HttpUrl
