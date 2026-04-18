"""Tests for hard filters + best-photo ranking in ``facade_grammar.pipeline.selection``."""

from datetime import UTC, datetime

from pydantic import HttpUrl
from shapely.geometry import LineString
from shapely.strtree import STRtree

from facade_grammar.config import SelectionConfig
from facade_grammar.geo import WGS84_TO_RD
from facade_grammar.pipeline.selection import _passes_hard_filters, _top_k_photos
from facade_grammar.schemas.buildings import Facade
from facade_grammar.schemas.photos import PhotoMetadata

_CFG = SelectionConfig()

# A canal facade running east along y = 0, outward normal pointing north (0 deg).
# All coords treated as already-RD (metres) for the pure-logic filters.
_FACADE_START_RD = (0.0, 0.0)
_FACADE_END_RD = (10.0, 0.0)
_FACADE_MID_RD = (5.0, 0.0)
_EDGE = LineString([_FACADE_START_RD, _FACADE_END_RD])
_NORMAL = (0.0, 1.0)      # points north (+y)
_NORMAL_DEG = 0.0         # compass azimuth


def _photo(
    photo_id: str = "mly-1",
    lon: float = 4.88,
    lat: float = 52.37,
    *,
    bearing: float | None = 180.0,
) -> PhotoMetadata:
    return PhotoMetadata(
        photo_id=photo_id,
        lon=lon,
        lat=lat,
        bearing_deg=bearing,
        captured_at=datetime(2024, 1, 1, tzinfo=UTC),
        url=HttpUrl("https://example.com/x.jpg"),
    )


def _run_filter(rd: tuple[float, float], *, bearing: float | None = 180.0) -> bool:
    return _passes_hard_filters(
        _photo(bearing=bearing), rd, _EDGE, _FACADE_MID_RD, _NORMAL, _NORMAL_DEG, _CFG
    )


def test_happy_path_passes() -> None:
    # 20m north of the facade midpoint, looking south back at it.
    assert _run_filter((5.0, 20.0), bearing=180.0)


def test_missing_bearing_rejected() -> None:
    assert not _run_filter((5.0, 20.0), bearing=None)


def test_too_close_rejected() -> None:
    assert not _run_filter((5.0, _CFG.photo_min_dist_m - 0.1))


def test_too_far_rejected() -> None:
    assert not _run_filter((5.0, _CFG.photo_max_dist_m + 0.1))


def test_wrong_side_rejected() -> None:
    # South of the facade (inside the building) fails the outside check.
    assert not _run_filter((5.0, -20.0))


def test_bad_bearing_rejected() -> None:
    # Looking north instead of south: off by 180deg, well outside tolerance.
    assert not _run_filter((5.0, 20.0), bearing=0.0)


def test_bearing_within_tolerance() -> None:
    assert _run_filter((5.0, 20.0), bearing=180.0 + _CFG.bearing_tol_deg - 0.1)
    assert not _run_filter((5.0, 20.0), bearing=180.0 + _CFG.bearing_tol_deg + 0.1)


def test_bearing_wrap_around() -> None:
    # Facade requires bearing ~180; 179 and 181 both pass; 1 and 359 both fail.
    assert _run_filter((5.0, 20.0), bearing=179.0)
    assert _run_filter((5.0, 20.0), bearing=181.0)
    assert not _run_filter((5.0, 20.0), bearing=1.0)
    assert not _run_filter((5.0, 20.0), bearing=359.0)


def _canal_facade() -> tuple[Facade, LineString, tuple[float, float]]:
    """Return (facade, rd-edge, rd-midpoint) so tests can place photos relative to it."""
    facade = Facade(
        building_id="b1",
        edge_start=(4.88, 52.37),
        edge_end=(4.881, 52.37),
        classification="canal",
        normal_deg=0.0,
    )
    sx, sy = WGS84_TO_RD.transform(4.88, 52.37)
    ex, ey = WGS84_TO_RD.transform(4.881, 52.37)
    return facade, LineString([(sx, sy), (ex, ey)]), ((sx + ex) / 2, (sy + ey) / 2)


def test_top_k_without_water_orders_by_distance() -> None:
    facade, edge, (mx, my) = _canal_facade()
    empty_tree = STRtree([])
    close = _photo("close", bearing=180.0)
    far = _photo("far", bearing=180.0)
    close_rd = (mx, my + 10.0)
    far_rd = (mx, my + 30.0)
    w1 = _top_k_photos(facade, edge, (mx, my), [close, far], [close_rd, far_rd], empty_tree, _CFG)
    w2 = _top_k_photos(facade, edge, (mx, my), [far, close], [far_rd, close_rd], empty_tree, _CFG)
    assert [p.photo_id for p in w1] == ["close", "far"]
    assert [p.photo_id for p in w2] == ["close", "far"]


def test_top_k_prefers_across_canal_over_same_side() -> None:
    facade, edge, (mx, my) = _canal_facade()
    # Canal strip east-west, north of the facade.
    canal = LineString([(mx - 100.0, my + 15.0), (mx + 100.0, my + 15.0)])
    tree = STRtree([canal])
    near_same_side = _photo("near-same", bearing=180.0)
    far_across = _photo("far-across", bearing=180.0)
    # near_rd sight line doesn't reach the canal strip at y=15;
    # far_rd sight line from (mx, my+30) to the midpoint crosses y=15.
    near_rd = (mx, my + 10.0)
    far_rd = (mx, my + 30.0)
    result = _top_k_photos(
        facade, edge, (mx, my), [near_same_side, far_across], [near_rd, far_rd], tree, _CFG
    )
    assert [p.photo_id for p in result] == ["far-across", "near-same"]


def test_top_k_returns_empty_when_all_filtered() -> None:
    facade, edge, (mx, my) = _canal_facade()
    no_bearing = _photo(bearing=None)
    result = _top_k_photos(
        facade, edge, (mx, my), [no_bearing], [(mx, my + 20.0)], STRtree([]), _CFG
    )
    assert result == []


def test_top_k_respects_k_limit() -> None:
    facade, edge, (mx, my) = _canal_facade()
    photos = [_photo(f"p{i}", bearing=180.0) for i in range(5)]
    photos_rd = [(mx, my + 10.0 + i) for i in range(5)]
    tight = _CFG.model_copy(update={"top_k": 2})
    result = _top_k_photos(facade, edge, (mx, my), photos, photos_rd, STRtree([]), tight)
    assert len(result) == 2
