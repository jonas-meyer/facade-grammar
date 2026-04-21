"""Mapillary photo metadata records."""

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, HttpUrl


class PhotoMetadata(BaseModel):
    """Single Mapillary photo with camera pose.

    ``bearing_deg`` is ``None`` when Mapillary has no SfM-refined heading for
    the image (its ``compass_angle`` comes back as ``-1``). For panos,
    ``bearing_deg`` represents the forward direction of the image — the
    column at the horizontal centre of the equirectangular frame.
    """

    model_config = ConfigDict(frozen=True)

    photo_id: str
    lon: float
    lat: float
    bearing_deg: float | None = Field(default=None, ge=0, lt=360)
    altitude_m: float | None = None
    captured_at: AwareDatetime
    is_pano: bool = False
    url: HttpUrl
    quality_score: float | None = Field(default=None, ge=0, le=1)
