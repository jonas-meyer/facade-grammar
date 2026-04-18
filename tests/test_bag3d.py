"""Tests for ``facade_grammar.clients.bag3d._BagBuilding``."""

from facade_grammar.clients.bag3d import _BagBuilding

_RD_RING = [
    (120000.0, 485000.0),
    (120010.0, 485000.0),
    (120010.0, 485010.0),
    (120000.0, 485010.0),
    (120000.0, 485000.0),
]


def _feature(**prop_overrides: float | str) -> dict[str, object]:
    props: dict[str, float | str] = {
        "identificatie": "NL.IMBAG.Pand.0363100000000001",
        "b3_h_max": 14.5,
        "b3_h_maaiveld": -0.5,
    }
    props.update(prop_overrides)
    return {
        "geometry": {"type": "Polygon", "coordinates": [_RD_RING]},
        "properties": props,
    }


def test_polygon_feature_to_building() -> None:
    building = _BagBuilding.model_validate(_feature()).to_building()
    assert building.building_id == "NL.IMBAG.Pand.0363100000000001"
    assert building.height_m == 15.0  # 14.5 minus -0.5
    assert building.ground_altitude_m == -0.5
    assert len(building.footprint) == len(_RD_RING)
    # Amsterdam-area RD coords reproject to ~(4.88 E, 52.37 N) in WGS84.
    lon0, lat0 = building.footprint[0]
    assert 4.84 < lon0 < 4.90
    assert 52.34 < lat0 < 52.40


def test_multipolygon_takes_first_ring() -> None:
    second_ring = [(x + 100, y + 100) for x, y in _RD_RING]
    feat = _feature()
    feat["geometry"] = {
        "type": "MultiPolygon",
        "coordinates": [[_RD_RING], [second_ring]],
    }
    building = _BagBuilding.model_validate(feat).to_building()
    lon0, lat0 = building.footprint[0]
    # First ring's first vertex corresponds to RD (120000, 485000), not the
    # shifted second ring.
    assert 4.84 < lon0 < 4.90
    assert 52.34 < lat0 < 52.40


def test_height_computed_from_bag_fields() -> None:
    building = _BagBuilding.model_validate(
        _feature(b3_h_max=20.25, b3_h_maaiveld=1.25)
    ).to_building()
    assert building.height_m == 19.0
