"""Geometry-first image localization utilities.

This module replaces the old "FOV + trapezoid interpolation" idea with the
chain used in the research note:

pixel -> undistorted camera ray -> camera pose -> ground plane / DEM / GCP model.

Coordinate convention:
    * Image pixel: (u, v), u right, v down.
    * Camera frame: x right, y down, z forward along the optical axis.
    * Local world frame: ENU meters, x east, y north, z up.
    * Rotation matrices named camera_to_enu have camera axes as columns.

The code intentionally keeps dependencies light. It only needs numpy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Literal, Sequence
import math

import numpy as np


EPS = 1e-12
WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)

BBoxAnchor = Literal[
    "center",
    "bottom_center",
    "top_center",
    "left_center",
    "right_center",
]


def _arr(values: Sequence[float] | np.ndarray, *, shape: tuple[int, ...] | None = None) -> np.ndarray:
    out = np.asarray(values, dtype=float)
    if shape is not None and out.shape != shape:
        raise ValueError(f"expected shape {shape}, got {out.shape}")
    return out


def normalize(vector: Sequence[float] | np.ndarray) -> np.ndarray:
    """Return a unit vector and fail loudly for near-zero inputs."""

    v = np.asarray(vector, dtype=float)
    n = float(np.linalg.norm(v))
    if n < EPS:
        raise ValueError("cannot normalize a near-zero vector")
    return v / n


def bbox_to_pixel(
    bbox_xyxy: Sequence[float],
    anchor: BBoxAnchor = "bottom_center",
) -> tuple[float, float]:
    """Convert a detection box to the image point that should be localized.

    The default is bottom_center because many framed objects are upright and
    the useful geographic point is their contact point with the ground. For
    flat ground objects such as road markings, use anchor="center".
    """

    x1, y1, x2, y2 = map(float, bbox_xyxy)
    if x2 < x1 or y2 < y1:
        raise ValueError("bbox must be [x_min, y_min, x_max, y_max]")

    if anchor == "center":
        return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)
    if anchor == "bottom_center":
        return ((x1 + x2) * 0.5, y2)
    if anchor == "top_center":
        return ((x1 + x2) * 0.5, y1)
    if anchor == "left_center":
        return (x1, (y1 + y2) * 0.5)
    if anchor == "right_center":
        return (x2, (y1 + y2) * 0.5)
    raise ValueError(f"unknown bbox anchor: {anchor}")


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole intrinsics with Brown-Conrady distortion.

    distortion order is (k1, k2, p1, p2, k3). Use zeros when unknown, but for
    production accuracy this should come from calibration.
    """

    fx: float
    fy: float
    cx: float
    cy: float
    distortion: Sequence[float] = (0.0, 0.0, 0.0, 0.0, 0.0)
    width: int | None = None
    height: int | None = None

    @property
    def k(self) -> np.ndarray:
        return np.array(
            [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )

    def normalized_pixel(self, pixel: Sequence[float]) -> np.ndarray:
        """Return distorted normalized camera coordinates."""

        u, v = map(float, pixel)
        return np.array([(u - self.cx) / self.fx, (v - self.cy) / self.fy], dtype=float)

    def distort_normalized(self, xy: Sequence[float]) -> np.ndarray:
        """Apply Brown-Conrady distortion to normalized coordinates."""

        x, y = map(float, xy)
        k1, k2, p1, p2, k3 = self._distortion5()
        r2 = x * x + y * y
        r4 = r2 * r2
        r6 = r4 * r2
        radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
        x_tan = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        y_tan = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
        return np.array([x * radial + x_tan, y * radial + y_tan], dtype=float)

    def undistort_pixel(self, pixel: Sequence[float], iterations: int = 8) -> np.ndarray:
        """Return undistorted normalized camera coordinates.

        OpenCV normally does this, but the workspace does not currently have
        cv2 installed. This iterative inverse is accurate enough for standard
        radial/tangential coefficients when calibration values are reasonable.
        """

        xd, yd = self.normalized_pixel(pixel)
        k1, k2, p1, p2, k3 = self._distortion5()
        if max(abs(k1), abs(k2), abs(p1), abs(p2), abs(k3)) < EPS:
            return np.array([xd, yd], dtype=float)

        x, y = xd, yd
        for _ in range(iterations):
            r2 = x * x + y * y
            r4 = r2 * r2
            r6 = r4 * r2
            radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
            if abs(radial) < EPS:
                raise ValueError("invalid distortion parameters: radial term is near zero")
            x_tan = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
            y_tan = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
            x = (xd - x_tan) / radial
            y = (yd - y_tan) / radial
        return np.array([x, y], dtype=float)

    def pixel_to_camera_ray(self, pixel: Sequence[float]) -> np.ndarray:
        """Convert a pixel to a unit ray in camera coordinates."""

        x, y = self.undistort_pixel(pixel)
        return normalize(np.array([x, y, 1.0], dtype=float))

    def _distortion5(self) -> tuple[float, float, float, float, float]:
        values = list(self.distortion)
        if len(values) > 5:
            raise ValueError("distortion must have at most 5 values: k1,k2,p1,p2,k3")
        values += [0.0] * (5 - len(values))
        return tuple(float(v) for v in values)  # type: ignore[return-value]


@dataclass(frozen=True)
class CameraPose:
    """Camera pose in local ENU coordinates."""

    center_enu: Sequence[float]
    camera_to_enu: Sequence[Sequence[float]]

    def __post_init__(self) -> None:
        center = _arr(self.center_enu, shape=(3,))
        rotation = _arr(self.camera_to_enu, shape=(3, 3))
        err = np.linalg.norm(rotation.T @ rotation - np.eye(3))
        det = float(np.linalg.det(rotation))
        if err > 1e-5 or abs(det - 1.0) > 1e-5:
            raise ValueError(
                "camera_to_enu must be a right-handed orthonormal rotation matrix"
            )
        object.__setattr__(self, "center_enu", center)
        object.__setattr__(self, "camera_to_enu", rotation)

    @classmethod
    def from_axes(
        cls,
        center_enu: Sequence[float],
        right_enu: Sequence[float],
        down_enu: Sequence[float],
        forward_enu: Sequence[float],
    ) -> "CameraPose":
        """Build a pose from the three camera axes expressed in ENU."""

        rotation = np.column_stack(
            [
                normalize(right_enu),
                normalize(down_enu),
                normalize(forward_enu),
            ]
        )
        rotation = orthonormalize_rotation(rotation)
        return cls(center_enu=center_enu, camera_to_enu=rotation)

    @classmethod
    def look_at(
        cls,
        center_enu: Sequence[float],
        target_enu: Sequence[float],
        image_up_hint_enu: Sequence[float] = (0.0, 1.0, 0.0),
    ) -> "CameraPose":
        """Point the camera optical axis from center to target.

        image_up_hint_enu is the desired direction of the top of the image.
        For a nadir camera with north-up images, keep the default.
        """

        center = _arr(center_enu, shape=(3,))
        target = _arr(target_enu, shape=(3,))
        forward = normalize(target - center)
        up_hint = normalize(image_up_hint_enu)
        right = np.cross(forward, up_hint)
        if np.linalg.norm(right) < 1e-8:
            fallback = np.array([1.0, 0.0, 0.0])
            right = np.cross(forward, fallback)
            if np.linalg.norm(right) < 1e-8:
                right = np.cross(forward, np.array([0.0, 1.0, 0.0]))
        right = normalize(right)
        down = normalize(np.cross(forward, right))
        return cls.from_axes(center, right, down, forward)

    @classmethod
    def nadir(
        cls,
        center_enu: Sequence[float],
        image_up_azimuth_deg: float = 0.0,
    ) -> "CameraPose":
        """Build a nadir pose.

        image_up_azimuth_deg is clockwise from north. 0 means image top points
        north and image right points east.
        """

        az = math.radians(image_up_azimuth_deg)
        image_up = np.array([math.sin(az), math.cos(az), 0.0])
        right = np.array([math.cos(az), -math.sin(az), 0.0])
        down = -image_up
        forward = np.array([0.0, 0.0, -1.0])
        return cls.from_axes(center_enu, right, down, forward)

    @classmethod
    def from_heading_tilt(
        cls,
        center_enu: Sequence[float],
        heading_deg: float,
        tilt_from_nadir_deg: float,
        roll_deg: float = 0.0,
    ) -> "CameraPose":
        """Build a ground-looking pose from optical-axis heading and tilt.

        heading_deg is clockwise from north. tilt_from_nadir_deg is 0 for a
        straight-down optical axis and approaches 90 near the horizon. roll_deg
        rotates the image clockwise around the optical axis as seen by camera.
        """

        heading = math.radians(float(heading_deg))
        tilt = math.radians(float(tilt_from_nadir_deg))
        if tilt < -EPS or tilt >= math.pi * 0.5:
            raise ValueError("tilt_from_nadir_deg must be in [0, 90)")

        forward = normalize(
            np.array(
                [
                    math.sin(heading) * math.sin(tilt),
                    math.cos(heading) * math.sin(tilt),
                    -math.cos(tilt),
                ],
                dtype=float,
            )
        )

        image_up_hint = np.array([0.0, 0.0, 1.0], dtype=float)
        if tilt < 1e-7:
            image_up_hint = np.array([math.sin(heading), math.cos(heading), 0.0])

        right = np.cross(forward, image_up_hint)
        if np.linalg.norm(right) < 1e-8:
            right = np.array([math.cos(heading), -math.sin(heading), 0.0])
        right = normalize(right)
        down = normalize(np.cross(forward, right))

        roll = math.radians(float(roll_deg))
        if abs(roll) > EPS:
            right, down = rotate_image_axes_about_forward(right, down, roll)

        return cls.from_axes(center_enu, right, down, forward)

    def world_ray(self, camera_ray: Sequence[float]) -> np.ndarray:
        """Transform a camera-frame ray to ENU."""

        return normalize(self.camera_to_enu @ _arr(camera_ray, shape=(3,)))


def orthonormalize_rotation(rotation: Sequence[Sequence[float]]) -> np.ndarray:
    """Project a near-rotation matrix onto SO(3)."""

    r = _arr(rotation, shape=(3, 3))
    u, _, vt = np.linalg.svd(r)
    out = u @ vt
    if np.linalg.det(out) < 0.0:
        u[:, -1] *= -1.0
        out = u @ vt
    return out


def rotate_image_axes_about_forward(
    right: Sequence[float],
    down: Sequence[float],
    roll_rad: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate camera image axes around the optical axis."""

    r = normalize(right)
    d = normalize(down)
    cos_roll = math.cos(float(roll_rad))
    sin_roll = math.sin(float(roll_rad))
    return (
        normalize(cos_roll * r + sin_roll * d),
        normalize(-sin_roll * r + cos_roll * d),
    )


def intersect_ray_z_plane(
    origin_enu: Sequence[float],
    direction_enu: Sequence[float],
    plane_z: float = 0.0,
    *,
    require_forward: bool = True,
) -> np.ndarray:
    """Intersect a 3D ray with the horizontal plane z=plane_z."""

    origin = _arr(origin_enu, shape=(3,))
    direction = normalize(direction_enu)
    if abs(direction[2]) < EPS:
        raise ValueError("ray is parallel to the z plane")
    distance = (float(plane_z) - origin[2]) / direction[2]
    if require_forward and distance < 0.0:
        raise ValueError("intersection is behind the camera")
    return origin + distance * direction


def pixel_to_world_ray(
    pixel: Sequence[float],
    intrinsics: CameraIntrinsics,
    pose: CameraPose,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (origin_enu, direction_enu) for one image pixel."""

    ray_camera = intrinsics.pixel_to_camera_ray(pixel)
    return np.asarray(pose.center_enu, dtype=float), pose.world_ray(ray_camera)


def locate_pixel_on_plane(
    pixel: Sequence[float],
    intrinsics: CameraIntrinsics,
    pose: CameraPose,
    plane_z: float = 0.0,
) -> np.ndarray:
    """Locate one pixel on a horizontal plane in local ENU meters."""

    origin, direction = pixel_to_world_ray(pixel, intrinsics, pose)
    return intersect_ray_z_plane(origin, direction, plane_z)


def locate_bbox_on_plane(
    bbox_xyxy: Sequence[float],
    intrinsics: CameraIntrinsics,
    pose: CameraPose,
    plane_z: float = 0.0,
    anchor: BBoxAnchor = "bottom_center",
) -> np.ndarray:
    """Locate a detection box anchor on a horizontal plane."""

    return locate_pixel_on_plane(
        bbox_to_pixel(bbox_xyxy, anchor=anchor),
        intrinsics,
        pose,
        plane_z=plane_z,
    )


def intersect_ray_dem(
    origin_enu: Sequence[float],
    direction_enu: Sequence[float],
    dem_height: Callable[[float, float], float],
    *,
    max_distance: float = 10000.0,
    initial_step: float = 1.0,
    growth: float = 1.6,
    tolerance: float = 0.02,
    max_iterations: int = 80,
) -> np.ndarray:
    """Intersect a ray with a DEM height function z=dem_height(x, y).

    The function brackets the first terrain crossing along the ray and then
    solves it by bisection. dem_height receives local ENU x/y meters.
    """

    origin = _arr(origin_enu, shape=(3,))
    direction = normalize(direction_enu)

    def signed_height(distance: float) -> float:
        p = origin + distance * direction
        return float(p[2]) - float(dem_height(float(p[0]), float(p[1])))

    low = 0.0
    f_low = signed_height(low)
    if abs(f_low) <= tolerance:
        return origin.copy()

    step = float(initial_step)
    high = step
    f_high = signed_height(high)
    while f_low * f_high > 0.0 and high < max_distance:
        low = high
        f_low = f_high
        step *= growth
        high = min(max_distance, high + step)
        f_high = signed_height(high)

    if f_low * f_high > 0.0:
        raise ValueError("ray did not intersect DEM within max_distance")

    for _ in range(max_iterations):
        mid = 0.5 * (low + high)
        f_mid = signed_height(mid)
        if abs(f_mid) <= tolerance or abs(high - low) <= tolerance:
            return origin + mid * direction
        if f_low * f_mid <= 0.0:
            high = mid
        else:
            low = mid
            f_low = f_mid
    return origin + 0.5 * (low + high) * direction


def locate_pixel_on_dem(
    pixel: Sequence[float],
    intrinsics: CameraIntrinsics,
    pose: CameraPose,
    dem_height: Callable[[float, float], float],
    **kwargs: float,
) -> np.ndarray:
    """Locate one pixel by ray/DEM intersection."""

    origin, direction = pixel_to_world_ray(pixel, intrinsics, pose)
    return intersect_ray_dem(origin, direction, dem_height, **kwargs)


@dataclass(frozen=True)
class RegularGridDEM:
    """Small bilinear DEM helper for local ENU grids.

    x0/y0 are the ENU coordinates of elevation[0, 0].
    dx/dy are grid spacing in meters. elevation shape is (rows_y, cols_x).
    """

    x0: float
    y0: float
    dx: float
    dy: float
    elevation: Sequence[Sequence[float]]
    nodata: float | None = None

    def __post_init__(self) -> None:
        z = np.asarray(self.elevation, dtype=float)
        if z.ndim != 2 or z.shape[0] < 2 or z.shape[1] < 2:
            raise ValueError("elevation must be a 2D array with at least 2x2 cells")
        if abs(self.dx) < EPS or abs(self.dy) < EPS:
            raise ValueError("dx and dy must be non-zero")
        object.__setattr__(self, "elevation", z)

    def height(self, x: float, y: float) -> float:
        col = (float(x) - self.x0) / self.dx
        row = (float(y) - self.y0) / self.dy
        c0 = math.floor(col)
        r0 = math.floor(row)
        if c0 < 0 or r0 < 0 or c0 + 1 >= self.elevation.shape[1] or r0 + 1 >= self.elevation.shape[0]:
            raise ValueError("point is outside the DEM grid")
        tx = col - c0
        ty = row - r0
        z00 = float(self.elevation[r0, c0])
        z10 = float(self.elevation[r0, c0 + 1])
        z01 = float(self.elevation[r0 + 1, c0])
        z11 = float(self.elevation[r0 + 1, c0 + 1])
        values = (z00, z10, z01, z11)
        if self.nodata is not None and any(v == self.nodata for v in values):
            raise ValueError("DEM interpolation touched nodata")
        return (
            z00 * (1.0 - tx) * (1.0 - ty)
            + z10 * tx * (1.0 - ty)
            + z01 * (1.0 - tx) * ty
            + z11 * tx * ty
        )


def _normalize_points_2d(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centroid = points.mean(axis=0)
    shifted = points - centroid
    mean_dist = float(np.mean(np.linalg.norm(shifted, axis=1)))
    scale = math.sqrt(2.0) / mean_dist if mean_dist > EPS else 1.0
    transform = np.array(
        [
            [scale, 0.0, -scale * centroid[0]],
            [0.0, scale, -scale * centroid[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    homogeneous = np.column_stack([points, np.ones(len(points))])
    normalized = (transform @ homogeneous.T).T[:, :2]
    return normalized, transform


def estimate_homography(
    src_points: Sequence[Sequence[float]],
    dst_points: Sequence[Sequence[float]],
) -> np.ndarray:
    """Estimate a 3x3 homography from src 2D points to dst 2D points.

    At least 4 non-collinear point pairs are required. The implementation uses
    normalized DLT, which is the same model used by planar GCP localization.
    """

    src = _arr(src_points)
    dst = _arr(dst_points)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 2:
        raise ValueError("src_points and dst_points must both be Nx2 arrays")
    if src.shape[0] < 4:
        raise ValueError("at least 4 point pairs are required")

    src_n, t_src = _normalize_points_2d(src)
    dst_n, t_dst = _normalize_points_2d(dst)
    rows = []
    for (x, y), (u, v) in zip(src_n, dst_n):
        rows.append([-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u])
        rows.append([0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v])
    a = np.asarray(rows, dtype=float)
    _, _, vt = np.linalg.svd(a)
    h_norm = vt[-1].reshape(3, 3)
    h = np.linalg.inv(t_dst) @ h_norm @ t_src
    if abs(h[2, 2]) > EPS:
        h = h / h[2, 2]
    return h


def apply_homography(
    homography: Sequence[Sequence[float]],
    points: Sequence[float] | Sequence[Sequence[float]] | np.ndarray,
) -> np.ndarray:
    """Apply a homography to one point or an Nx2 array."""

    h = _arr(homography, shape=(3, 3))
    pts = np.asarray(points, dtype=float)
    single = pts.ndim == 1
    if single:
        pts = pts.reshape(1, 2)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points must be a 2-vector or an Nx2 array")
    homogeneous = np.column_stack([pts, np.ones(len(pts))])
    mapped = (h @ homogeneous.T).T
    denom = mapped[:, 2:3]
    if np.any(np.abs(denom) < EPS):
        raise ValueError("homography mapped point to infinity")
    out = mapped[:, :2] / denom
    return out[0] if single else out


@dataclass(frozen=True)
class PlanarHomographyLocalizer:
    """Localize image points onto one known plane using planar GCPs."""

    homography: Sequence[Sequence[float]]
    plane_z: float = 0.0
    intrinsics: CameraIntrinsics | None = None
    coordinates_are_undistorted: bool = False

    @classmethod
    def from_gcps(
        cls,
        image_points: Sequence[Sequence[float]],
        ground_xy: Sequence[Sequence[float]],
        *,
        intrinsics: CameraIntrinsics | None = None,
        plane_z: float = 0.0,
        undistort: bool = True,
    ) -> "PlanarHomographyLocalizer":
        """Build an image->ground localizer from planar GCP pairs.

        If intrinsics is provided and undistort=True, the homography is fitted
        from undistorted normalized image coordinates to ENU x/y meters.
        Otherwise it is fitted directly from raw pixel coordinates.
        """

        img = _arr(image_points)
        if img.ndim != 2 or img.shape[1] != 2:
            raise ValueError("image_points must be an Nx2 array")
        if intrinsics is not None and undistort:
            src = np.array([intrinsics.undistort_pixel(p) for p in img], dtype=float)
            coords_undistorted = True
        else:
            src = img
            coords_undistorted = False
        h = estimate_homography(src, ground_xy)
        return cls(
            homography=h,
            plane_z=float(plane_z),
            intrinsics=intrinsics,
            coordinates_are_undistorted=coords_undistorted,
        )

    def locate_pixel(self, pixel: Sequence[float]) -> np.ndarray:
        if self.coordinates_are_undistorted:
            if self.intrinsics is None:
                raise ValueError("intrinsics are required for undistorted homography")
            src = self.intrinsics.undistort_pixel(pixel)
        else:
            src = _arr(pixel, shape=(2,))
        xy = apply_homography(self.homography, src)
        return np.array([xy[0], xy[1], self.plane_z], dtype=float)

    def locate_bbox(
        self,
        bbox_xyxy: Sequence[float],
        anchor: BBoxAnchor = "bottom_center",
    ) -> np.ndarray:
        return self.locate_pixel(bbox_to_pixel(bbox_xyxy, anchor=anchor))


def estimate_pose_dlt(
    object_points_enu: Sequence[Sequence[float]],
    image_points: Sequence[Sequence[float]],
    intrinsics: CameraIntrinsics,
) -> CameraPose:
    """Estimate camera pose from 3D-2D GCPs using linear DLT.

    Use this as an initialization/check path. For production PnP, refine the
    result with nonlinear reprojection minimization or OpenCV solvePnP when cv2
    is available. The points should be non-coplanar and expressed in local ENU
    meters.
    """

    obj = _arr(object_points_enu)
    img = _arr(image_points)
    if obj.ndim != 2 or obj.shape[1] != 3:
        raise ValueError("object_points_enu must be an Nx3 array")
    if img.ndim != 2 or img.shape != (obj.shape[0], 2):
        raise ValueError("image_points must be an Nx2 array matching object points")
    if obj.shape[0] < 6:
        raise ValueError("DLT pose estimation requires at least 6 3D-2D pairs")

    xy = np.array([intrinsics.undistort_pixel(p) for p in img], dtype=float)
    xh = np.column_stack([obj, np.ones(len(obj))])
    rows = []
    for point_h, (x, y) in zip(xh, xy):
        rows.append([*point_h, 0.0, 0.0, 0.0, 0.0, *(-x * point_h)])
        rows.append([0.0, 0.0, 0.0, 0.0, *point_h, *(-y * point_h)])
    a = np.asarray(rows, dtype=float)
    _, _, vt = np.linalg.svd(a)
    p = vt[-1].reshape(3, 4)
    m = p[:, :3]
    t = p[:, 3]
    if np.linalg.det(m) < 0.0:
        p *= -1.0
        m = p[:, :3]
        t = p[:, 3]
    scale = float(np.mean(np.linalg.norm(m, axis=1)))
    if scale < EPS:
        raise ValueError("degenerate DLT pose solution")
    world_to_camera = orthonormalize_rotation(m / scale)
    translation = t / scale
    center = -(world_to_camera.T @ translation)
    camera_to_enu = world_to_camera.T
    return CameraPose(center_enu=center, camera_to_enu=camera_to_enu)


@dataclass(frozen=True)
class LocalTangentPlane:
    """WGS84 geodetic <-> local ENU transformer.

    This avoids pyproj for local engineering work. For national grid outputs,
    authoritative datum shifts, geoid models, or epoch handling, use PROJ/pyproj
    in the application layer and keep this ENU frame as the numeric work frame.
    """

    origin_lat_deg: float
    origin_lon_deg: float
    origin_h_m: float = 0.0

    def __post_init__(self) -> None:
        origin = geodetic_to_ecef(self.origin_lat_deg, self.origin_lon_deg, self.origin_h_m)
        rotation = ecef_to_enu_rotation(self.origin_lat_deg, self.origin_lon_deg)
        object.__setattr__(self, "_origin_ecef", origin)
        object.__setattr__(self, "_ecef_to_enu", rotation)
        object.__setattr__(self, "_enu_to_ecef", rotation.T)

    def geodetic_to_enu(self, lat_deg: float, lon_deg: float, h_m: float = 0.0) -> np.ndarray:
        ecef = geodetic_to_ecef(lat_deg, lon_deg, h_m)
        return self._ecef_to_enu @ (ecef - self._origin_ecef)

    def enu_to_geodetic(self, enu: Sequence[float]) -> tuple[float, float, float]:
        ecef = self._origin_ecef + self._enu_to_ecef @ _arr(enu, shape=(3,))
        return ecef_to_geodetic(ecef)


def geodetic_to_ecef(lat_deg: float, lon_deg: float, h_m: float = 0.0) -> np.ndarray:
    lat = math.radians(float(lat_deg))
    lon = math.radians(float(lon_deg))
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    sin_lon = math.sin(lon)
    cos_lon = math.cos(lon)
    n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    x = (n + h_m) * cos_lat * cos_lon
    y = (n + h_m) * cos_lat * sin_lon
    z = (n * (1.0 - WGS84_E2) + h_m) * sin_lat
    return np.array([x, y, z], dtype=float)


def ecef_to_geodetic(ecef: Sequence[float]) -> tuple[float, float, float]:
    x, y, z = map(float, ecef)
    lon = math.atan2(y, x)
    p = math.hypot(x, y)
    lat = math.atan2(z, p * (1.0 - WGS84_E2))
    h = 0.0
    for _ in range(10):
        sin_lat = math.sin(lat)
        n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
        h = p / max(math.cos(lat), EPS) - n
        next_lat = math.atan2(z, p * (1.0 - WGS84_E2 * n / (n + h)))
        if abs(next_lat - lat) < 1e-13:
            lat = next_lat
            break
        lat = next_lat
    return math.degrees(lat), math.degrees(lon), h


def ecef_to_enu_rotation(lat_deg: float, lon_deg: float) -> np.ndarray:
    lat = math.radians(float(lat_deg))
    lon = math.radians(float(lon_deg))
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    sin_lon = math.sin(lon)
    cos_lon = math.cos(lon)
    return np.array(
        [
            [-sin_lon, cos_lon, 0.0],
            [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
            [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat],
        ],
        dtype=float,
    )


def reprojection_error(
    pose: CameraPose,
    intrinsics: CameraIntrinsics,
    object_points_enu: Sequence[Sequence[float]],
    image_points: Sequence[Sequence[float]],
) -> np.ndarray:
    """Return pixel residuals for diagnostics."""

    obj = _arr(object_points_enu)
    img = _arr(image_points)
    if obj.ndim != 2 or obj.shape[1] != 3:
        raise ValueError("object_points_enu must be Nx3")
    if img.shape != (obj.shape[0], 2):
        raise ValueError("image_points must be Nx2 matching object points")

    world_to_camera = np.asarray(pose.camera_to_enu, dtype=float).T
    center = np.asarray(pose.center_enu, dtype=float)
    residuals = []
    for point, observed in zip(obj, img):
        camera_xyz = world_to_camera @ (point - center)
        if camera_xyz[2] <= EPS:
            residuals.append([math.nan, math.nan])
            continue
        x = camera_xyz[0] / camera_xyz[2]
        y = camera_xyz[1] / camera_xyz[2]
        xd, yd = intrinsics.distort_normalized((x, y))
        projected = np.array(
            [intrinsics.fx * xd + intrinsics.cx, intrinsics.fy * yd + intrinsics.cy],
            dtype=float,
        )
        residuals.append(projected - observed)
    return np.asarray(residuals, dtype=float)


def rmse(values: Sequence[Sequence[float]] | np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return math.nan
    return float(math.sqrt(np.mean(arr * arr)))
