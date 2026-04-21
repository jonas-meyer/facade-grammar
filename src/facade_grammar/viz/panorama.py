"""Equirectangular-to-rectilinear reprojection for Mapillary 360 panoramas.

Convention: a pano whose ``compass_angle = 0`` has its *centre column* pointing
compass-north; column offset grows clockwise (east at W/4, south at W/2-right
of centre, west at W/4-left of centre — i.e., azimuth increases left-to-right).
Elevation runs top-to-bottom from +90 (up) to -90 (down).

``rectilinear_view`` extracts a perspective window oriented at a target compass
bearing. ``world_to_pixel`` is the inverse: given a WGS84 point, where would it
land in a view extracted at a given yaw/FoV — used to target SAM instance
picking onto the specific building we care about. All math is numpy + pyproj;
no OpenCV / torch.
"""

import functools
import io
import math

import numpy as np
from PIL import Image

from facade_grammar.geo import WGS84_GEOD


def rectilinear_view(
    pano_bytes: bytes,
    *,
    yaw_deg: float,
    pano_yaw_deg: float = 0.0,
    pitch_deg: float = 0.0,
    fov_deg: float = 90.0,
    size_px: tuple[int, int] = (1024, 768),
) -> bytes:
    """Reproject an equirectangular pano into a perspective view.

    ``yaw_deg`` is the target compass bearing (0 = north, 90 = east).
    ``pano_yaw_deg`` is Mapillary's ``compass_angle`` for the source pano.
    """
    pano = np.asarray(Image.open(io.BytesIO(pano_bytes)).convert("RGB"))
    pano_h, pano_w = pano.shape[:2]
    rays = _camera_rays(size_px, fov_deg)

    yaw_rel = np.deg2rad(yaw_deg - pano_yaw_deg)
    pitch = np.deg2rad(pitch_deg)
    cos_y, sin_y = np.cos(yaw_rel), np.sin(yaw_rel)
    cos_p, sin_p = np.cos(pitch), np.sin(pitch)
    rot = np.array(
        [
            [cos_y, sin_y * sin_p, sin_y * cos_p],
            [0.0, cos_p, -sin_p],
            [-sin_y, cos_y * sin_p, cos_y * cos_p],
        ]
    )
    world = rays @ rot.T

    azimuth = np.arctan2(world[..., 0], world[..., 2])
    elevation = np.arcsin(np.clip(-world[..., 1], -1.0, 1.0))
    u = (azimuth / (2 * np.pi) + 0.5) * pano_w
    v = (0.5 - elevation / np.pi) * pano_h

    buf = io.BytesIO()
    Image.fromarray(_bilinear_sample(pano, u, v)).save(buf, format="JPEG", quality=90)
    return buf.getvalue()


@functools.lru_cache(maxsize=8)
def _camera_rays(size_px: tuple[int, int], fov_deg: float) -> np.ndarray:
    """Unit-length camera-frame ray directions for every output pixel.

    Cached per ``(size, fov)`` since the grid is identical across reprojections
    with the same view parameters — only the rotation changes per call.
    """
    out_w, out_h = size_px
    fov = np.deg2rad(fov_deg)
    focal = out_w / (2 * np.tan(fov / 2))
    xs = np.arange(out_w) - (out_w - 1) / 2
    ys = np.arange(out_h) - (out_h - 1) / 2
    xx, yy = np.meshgrid(xs, ys)
    rays = np.stack([xx, yy, np.full_like(xx, focal)], axis=-1)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    return rays


def world_to_pixel(
    *,
    photo_lon: float,
    photo_lat: float,
    camera_z_m: float,
    world_lon: float,
    world_lat: float,
    world_z_m: float,
    view_yaw_deg: float,
    view_pitch_deg: float = 0.0,
    fov_deg: float,
    size_px: tuple[int, int],
) -> tuple[float, float] | None:
    """Project a WGS84 world point into a rectilinear view's pixel coords.

    ``camera_z_m`` and ``world_z_m`` must share the same vertical reference.
    ``view_pitch_deg`` must match the pitch used in ``rectilinear_view`` so the
    pixel coords line up. Returns ``None`` if the point is behind the camera.
    """
    azimuth, _, distance_m = WGS84_GEOD.inv(photo_lon, photo_lat, world_lon, world_lat)
    if distance_m < 1e-6:
        return None
    elevation_rad = math.atan2(world_z_m - camera_z_m, distance_m)
    yaw_rel_rad = math.radians(((azimuth - view_yaw_deg) + 180) % 360 - 180)
    pitch_rad = math.radians(view_pitch_deg)
    # Direction in the view-aimed frame (camera looks along +z), before applying pitch:
    cos_e = math.cos(elevation_rad)
    wx = math.sin(yaw_rel_rad) * cos_e
    wy = -math.sin(elevation_rad)
    wz = math.cos(yaw_rel_rad) * cos_e
    # Rotate -pitch around the x-axis to transform into camera frame (pinhole):
    cos_p, sin_p = math.cos(pitch_rad), math.sin(pitch_rad)
    ray_y = cos_p * wy + sin_p * wz
    ray_z = -sin_p * wy + cos_p * wz
    if ray_z <= 1e-9:
        return None
    fov_rad = math.radians(fov_deg)
    focal = size_px[0] / (2 * math.tan(fov_rad / 2))
    x = wx / ray_z * focal + size_px[0] / 2
    y = ray_y / ray_z * focal + size_px[1] / 2
    return x, y


def _bilinear_sample(img: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    u = u % w  # wrap horizontally
    v = np.clip(v, 0, h - 1)
    u0 = np.floor(u).astype(np.int64) % w
    u1 = (u0 + 1) % w
    v0 = np.floor(v).astype(np.int64)
    v1 = np.clip(v0 + 1, 0, h - 1)
    du = (u - u0)[..., None]
    dv = (v - v0)[..., None]
    top = img[v0, u0] * (1 - du) + img[v0, u1] * du
    bot = img[v1, u0] * (1 - du) + img[v1, u1] * du
    return np.clip(top * (1 - dv) + bot * dv, 0, 255).astype(np.uint8)
