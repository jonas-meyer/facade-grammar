"""Tests for the pinhole projection math in ``facade_grammar.viz.panorama``."""

from facade_grammar.viz.panorama import world_to_pixel

_AT = {"photo_lon": 4.9, "photo_lat": 52.37, "camera_z_m": 0.0}
_NORTH_100M = {"world_lon": 4.9, "world_lat": 52.371}
_VIEW = {"view_yaw_deg": 0.0, "fov_deg": 45.0, "size_px": (1280, 960)}


def test_forward_target_lands_at_view_centre() -> None:
    """Target directly ahead at the same elevation projects to the image centre."""
    res = world_to_pixel(**_AT, **_NORTH_100M, world_z_m=0.0, view_pitch_deg=0.0, **_VIEW)
    assert res is not None
    x, y = res
    assert abs(x - 640) < 0.5
    assert abs(y - 480) < 0.5


def test_target_east_of_view_axis_projects_to_right_half() -> None:
    res = world_to_pixel(
        **_AT,
        world_lon=4.901,
        world_lat=52.371,
        world_z_m=0.0,
        view_pitch_deg=0.0,
        **_VIEW,
    )
    assert res is not None
    x, _ = res
    assert x > 640


def test_high_target_projects_above_centre() -> None:
    """Roof-height target at same ground position has y < h/2 (above centre)."""
    res = world_to_pixel(**_AT, **_NORTH_100M, world_z_m=10.0, view_pitch_deg=0.0, **_VIEW)
    assert res is not None
    _, y = res
    assert y < 480


def test_pitch_up_lowers_horizon_level_target() -> None:
    """Pitching the view up by 10 deg pushes an elevation-0 target below centre."""
    res = world_to_pixel(**_AT, **_NORTH_100M, world_z_m=0.0, view_pitch_deg=10.0, **_VIEW)
    assert res is not None
    _, y = res
    assert y > 480


def test_target_behind_camera_returns_none() -> None:
    res = world_to_pixel(
        **_AT,
        world_lon=4.9,
        world_lat=52.369,
        world_z_m=0.0,
        view_pitch_deg=0.0,
        **_VIEW,
    )
    assert res is None


def test_coincident_photo_and_target_returns_none() -> None:
    res = world_to_pixel(
        **_AT,
        world_lon=4.9,
        world_lat=52.37,
        world_z_m=0.0,
        view_pitch_deg=0.0,
        **_VIEW,
    )
    assert res is None
