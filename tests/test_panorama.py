"""Tests for the pinhole projection math in ``facade_grammar.viz.panorama``."""

from functools import partial

from facade_grammar.viz.panorama import world_to_pixel

# Baseline: camera at (4.9, 52.37, 0), target 100m north at ground level, view
# facing north at 45° FoV, 1280x960. Each test overrides just what it varies.
_project = partial(
    world_to_pixel,
    photo_lon=4.9,
    photo_lat=52.37,
    camera_z_m=0.0,
    world_lon=4.9,
    world_lat=52.371,
    world_z_m=0.0,
    view_yaw_deg=0.0,
    view_pitch_deg=0.0,
    fov_deg=45.0,
    size_px=(1280, 960),
)


def test_forward_target_lands_at_view_centre() -> None:
    """Target directly ahead at the same elevation projects to the image centre."""
    res = _project()
    assert res is not None
    x, y = res
    assert abs(x - 640) < 0.5
    assert abs(y - 480) < 0.5


def test_target_east_of_view_axis_projects_to_right_half() -> None:
    res = _project(world_lon=4.901)
    assert res is not None
    x, _ = res
    assert x > 640


def test_high_target_projects_above_centre() -> None:
    """Roof-height target at same ground position has y < h/2 (above centre)."""
    res = _project(world_z_m=10.0)
    assert res is not None
    _, y = res
    assert y < 480


def test_pitch_up_lowers_horizon_level_target() -> None:
    """Pitching the view up by 10 deg pushes an elevation-0 target below centre."""
    res = _project(view_pitch_deg=10.0)
    assert res is not None
    _, y = res
    assert y > 480


def test_target_behind_camera_returns_none() -> None:
    assert _project(world_lat=52.369) is None


def test_coincident_photo_and_target_returns_none() -> None:
    assert _project(world_lat=52.37) is None
