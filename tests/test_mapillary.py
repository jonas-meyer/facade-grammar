"""Tests for the pure-logic helpers in ``facade_grammar.clients.mapillary``."""

from datetime import UTC, datetime

from facade_grammar.clients.mapillary import _halve, _MapillaryPhoto, _tile_bbox
from facade_grammar.config import Bbox


def _item(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "id": "mly-123",
        "computed_geometry": {"type": "Point", "coordinates": [4.88, 52.37]},
        "compass_angle": 42.0,
        "captured_at": 1_700_000_000_000,
        "thumb_1024_url": "https://example.com/t.jpg",
    }
    base.update(overrides)
    return base


def test_happy_path() -> None:
    photo = _MapillaryPhoto.model_validate(_item()).to_photo_metadata()
    assert photo is not None
    assert photo.photo_id == "mly-123"
    assert (photo.lon, photo.lat) == (4.88, 52.37)
    assert photo.bearing_deg == 42.0
    assert photo.captured_at == datetime.fromtimestamp(1_700_000_000, tz=UTC)


def test_missing_geometry_skipped() -> None:
    photo = _MapillaryPhoto.model_validate(_item(computed_geometry=None)).to_photo_metadata()
    assert photo is None


def test_negative_compass_maps_to_none() -> None:
    photo = _MapillaryPhoto.model_validate(_item(compass_angle=-1.0)).to_photo_metadata()
    assert photo is not None
    assert photo.bearing_deg is None


def test_null_compass_maps_to_none() -> None:
    photo = _MapillaryPhoto.model_validate(_item(compass_angle=None)).to_photo_metadata()
    assert photo is not None
    assert photo.bearing_deg is None


def test_tile_bbox_produces_grid_squared_tiles() -> None:
    tiles = list(_tile_bbox(Bbox(0.0, 0.0, 4.0, 8.0), grid=2))
    assert len(tiles) == 4
    assert tiles[0] == Bbox(0.0, 0.0, 2.0, 4.0)
    assert tiles[-1] == Bbox(2.0, 4.0, 4.0, 8.0)
    for t in tiles:
        assert t.min_lon >= 0.0 and t.max_lon <= 4.0
        assert t.min_lat >= 0.0 and t.max_lat <= 8.0


def test_halve_produces_four_quadrants_covering_input() -> None:
    parent = Bbox(0.0, 0.0, 10.0, 10.0)
    subs = list(_halve(parent))
    assert len(subs) == 4
    total_area = sum((s.max_lon - s.min_lon) * (s.max_lat - s.min_lat) for s in subs)
    assert total_area == 100.0
    for s in subs:
        assert (s.max_lon - s.min_lon) == 5.0
        assert (s.max_lat - s.min_lat) == 5.0
