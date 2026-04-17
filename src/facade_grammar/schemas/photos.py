"""Mapillary photo metadata records."""

from datetime import datetime

from pydantic import BaseModel, Field, HttpUrl


class PhotoMetadata(BaseModel):
    """Single Mapillary photo with camera pose."""

    photo_id: str
    lon: float
    lat: float
    bearing_deg: float = Field(ge=0, lt=360)
    captured_at: datetime
    url: HttpUrl
