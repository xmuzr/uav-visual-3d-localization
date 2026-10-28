"""Optional multi-photo localization helpers.

This module is deliberately additive: it does not change the existing
single-photo AGL/LRF workflow.  It adds two independent building blocks:

* multi-view ray intersection for the same target clicked in several photos;
* local height-plane fitting from several trusted LRF hit points.

All calculations stay in the same local ENU meter frame used by
image_localization.py.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import numpy as np

from image_localization import (
    CameraIntrinsics,
    CameraPose,
    LocalTangentPlane,
    normalize,
    pixel_to_world_ray,
)


EPS = 1e-12


@dataclass(frozen=True)
class RayObservation:
    """One photo observation of the same physical target.

    origin_enu is the camera center. direction_enu is the world-frame ray
    passing through the user-clicked pixel.
    """

    image_id: str
    origin_enu: Sequence[float]
    direction_enu: Sequence[float]
    pixel: Sequence[float] | None = None
    weight: float = 1.0
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        origin = np.asarray(self.origin_enu, dtype=float)
        if origin.shape != (3,):
            raise ValueError("origin_enu must be a 3-vector")
        direction = normalize(self.direction_enu)
        if not np.isfinite(origin).all() or not np.isfinite(direction).all():
            raise ValueError("ray observation contains non-finite values")
        if self.weight <= 0.0:
            raise ValueError("ray observation weight must be positive")
        object.__setattr__(self, "origin_enu", origin)
        object.__setattr__(self, "direction_enu", direction)
        if self.pixel is not None:
            pixel = np.asarray(self.pixel, dtype=float)
            if pixel.shape != (2,):
                raise ValueError("pixel must be a 2-vector")
            object.__setattr__(self, "pixel", pixel)


@dataclass(frozen=True)
class TriangulationResult:
    """Least-squares 3D point estimated from multiple image rays."""

    point_enu: np.ndarray
    ray_count: int
    residuals_m: list[float]
    along_ray_m: list[float]
    residual_rmse_m: float
    residual_mean_m: float
    residual_max_m: float
    min_ray_angle_deg: float
    normal_matrix_condition: float
    behind_observation_count: int
    quality_level: str
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "point_enu": self.point_enu.tolist(),
            "ray_count": self.ray_count,
            "residuals_m": self.residuals_m,
            "along_ray_m": self.along_ray_m,
            "residual_rmse_m": self.residual_rmse_m,
            "residual_mean_m": self.residual_mean_m,
            "residual_max_m": self.residual_max_m,
            "min_ray_angle_deg": self.min_ray_angle_deg,
            "normal_matrix_condition": self.normal_matrix_condition,
            "behind_observation_count": self.behind_observation_count,
            "quality_level": self.quality_level,
            "warnings": self.warnings,
        }


def ray_observation_from_pixel(
    *,
    image_id: str,
    pixel: Sequence[float],
    intrinsics: CameraIntrinsics,
    pose: CameraPose,
    weight: float = 1.0,
    metadata: dict[str, Any] | None = None,
) -> RayObservation:
    """Build a RayObservation using the existing pixel->world-ray chain."""

    origin, direction = pixel_to_world_ray(pixel, intrinsics, pose)
    return RayObservation(
        image_id=image_id,
        origin_enu=origin,
        direction_enu=direction,
        pixel=pixel,
        weight=weight,
        metadata=metadata,
    )


def triangulate_rays(observations: Sequence[RayObservation]) -> TriangulationResult:
    """Estimate the 3D point closest to all observation rays.

    The solution minimizes the sum of squared perpendicular distances from the
    point to each ray:

        sum ||(I - d d^T) (P - C)||^2

    where C is the camera center and d is the unit ray direction.
    """

    rays = list(observations)
    if len(rays) < 2:
        raise ValueError("multi-view triangulation requires at least two observations")

    identity = np.eye(3)
    normal = np.zeros((3, 3), dtype=float)
    rhs = np.zeros(3, dtype=float)
    for obs in rays:
        origin = np.asarray(obs.origin_enu, dtype=float)
        direction = np.asarray(obs.direction_enu, dtype=float)
        projector = identity - np.outer(direction, direction)
        normal += obs.weight * projector
        rhs += obs.weight * (projector @ origin)

    condition = float(np.linalg.cond(normal))
    point = _solve_normal_equation(normal, rhs)
    residuals, along_ray = _ray_residuals(point, rays)
    angles = pairwise_ray_angles_deg(rays)
    min_angle = min(angles) if angles else math.nan
    behind_count = sum(1 for value in along_ray if value < 0.0)
    residual_rmse = _rmse_1d(residuals)
    residual_mean = float(np.mean(residuals)) if residuals else math.nan
    residual_max = float(max(residuals)) if residuals else math.nan
    warnings = _triangulation_warnings(
        ray_count=len(rays),
        min_angle_deg=min_angle,
        residual_rmse_m=residual_rmse,
        residual_max_m=residual_max,
        condition=condition,
        behind_count=behind_count,
    )
    quality = estimate_triangulation_quality(
        ray_count=len(rays),
        min_angle_deg=min_angle,
        residual_rmse_m=residual_rmse,
        residual_max_m=residual_max,
        condition=condition,
        behind_count=behind_count,
    )
    return TriangulationResult(
        point_enu=point,
        ray_count=len(rays),
        residuals_m=[float(v) for v in residuals],
        along_ray_m=[float(v) for v in along_ray],
        residual_rmse_m=residual_rmse,
        residual_mean_m=residual_mean,
        residual_max_m=residual_max,
        min_ray_angle_deg=float(min_angle),
        normal_matrix_condition=condition,
        behind_observation_count=behind_count,
        quality_level=quality,
        warnings=warnings,
    )


def _solve_normal_equation(normal: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.solve(normal, rhs)
    except np.linalg.LinAlgError:
        point, *_ = np.linalg.lstsq(normal, rhs, rcond=None)
        return point


def _ray_residuals(
    point: np.ndarray,
    observations: Sequence[RayObservation],
) -> tuple[list[float], list[float]]:
    residuals: list[float] = []
    along_ray: list[float] = []
    for obs in observations:
        origin = np.asarray(obs.origin_enu, dtype=float)
        direction = np.asarray(obs.direction_enu, dtype=float)
        delta = point - origin
        distance_along = float(np.dot(delta, direction))
        closest = origin + distance_along * direction
        residuals.append(float(np.linalg.norm(point - closest)))
        along_ray.append(distance_along)
    return residuals, along_ray


def pairwise_ray_angles_deg(observations: Sequence[RayObservation]) -> list[float]:
    """Return acute pairwise angles between all ray directions."""

    rays = list(observations)
    angles: list[float] = []
    for i in range(len(rays)):
        for j in range(i + 1, len(rays)):
            a = np.asarray(rays[i].direction_enu, dtype=float)
            b = np.asarray(rays[j].direction_enu, dtype=float)
            dot = float(np.clip(abs(np.dot(a, b)), 0.0, 1.0))
            angles.append(math.degrees(math.acos(dot)))
    return angles


def estimate_triangulation_quality(
    *,
    ray_count: int,
    min_angle_deg: float,
    residual_rmse_m: float,
    residual_max_m: float,
    condition: float,
    behind_count: int,
) -> str:
    """Conservative quality label for UI diagnostics."""

    if behind_count > 0 or not np.isfinite(condition) or condition > 1e10:
        return "low"
    if ray_count >= 3 and min_angle_deg >= 5.0 and residual_rmse_m <= 1.0 and residual_max_m <= 2.0:
        return "high"
    if ray_count >= 2 and min_angle_deg >= 1.0 and residual_rmse_m <= 5.0 and residual_max_m <= 10.0:
        return "medium"
    return "low"


def _triangulation_warnings(
    *,
    ray_count: int,
    min_angle_deg: float,
    residual_rmse_m: float,
    residual_max_m: float,
    condition: float,
    behind_count: int,
) -> list[str]:
    warnings: list[str] = []
    if ray_count < 3:
        warnings.append("only two observations; add a third photo when possible")
    if not np.isfinite(condition) or condition > 1e10:
        warnings.append("ray geometry is nearly singular")
    if np.isfinite(min_angle_deg) and min_angle_deg < 1.0:
        warnings.append("ray intersection angle is too small")
    if residual_rmse_m > 5.0 or residual_max_m > 10.0:
        warnings.append("ray residuals are large; check clicked points and pose metadata")
    if behind_count > 0:
        warnings.append("estimated point is behind at least one camera")
    return warnings


@dataclass(frozen=True)
class LrfPoint:
    """One trusted LRF hit point in local ENU meters."""

    id: str
    enu: Sequence[float]
    source_image_id: str | None = None
    distance_residual_m: float | None = None
    weight: float = 1.0
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        enu = np.asarray(self.enu, dtype=float)
        if enu.shape != (3,):
            raise ValueError("LrfPoint.enu must be a 3-vector")
        if not np.isfinite(enu).all():
            raise ValueError("LrfPoint.enu contains non-finite values")
        if self.weight <= 0.0:
            raise ValueError("LrfPoint.weight must be positive")
        object.__setattr__(self, "enu", enu)


@dataclass(frozen=True)
class HeightPlane:
    """Local height model z = a*x + b*y + c."""

    a: float
    b: float
    c: float
    point_count: int
    inlier_ids: list[str]
    rejected_ids: list[str]
    residuals_m: list[float]
    rmse_m: float
    max_abs_residual_m: float
    xy_span_m: float
    design_condition: float
    quality_level: str
    warnings: list[str]

    def height_at(self, x: float, y: float) -> float:
        return self.a * float(x) + self.b * float(y) + self.c

    @property
    def normal_enu(self) -> np.ndarray:
        return normalize(np.array([-self.a, -self.b, 1.0], dtype=float))

    def intersect_ray(
        self,
        origin_enu: Sequence[float],
        direction_enu: Sequence[float],
        *,
        require_forward: bool = True,
    ) -> np.ndarray:
        origin = np.asarray(origin_enu, dtype=float)
        direction = normalize(direction_enu)
        denominator = direction[2] - self.a * direction[0] - self.b * direction[1]
        if abs(denominator) < EPS:
            raise ValueError("ray is parallel to the fitted height plane")
        numerator = self.a * origin[0] + self.b * origin[1] + self.c - origin[2]
        distance = float(numerator / denominator)
        if require_forward and distance < 0.0:
            raise ValueError("height-plane intersection is behind the camera")
        return origin + distance * direction

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": "z = a*x + b*y + c",
            "a": self.a,
            "b": self.b,
            "c": self.c,
            "normal_enu": self.normal_enu.tolist(),
            "point_count": self.point_count,
            "inlier_count": len(self.inlier_ids),
            "inlier_ids": self.inlier_ids,
            "rejected_ids": self.rejected_ids,
            "residuals_m": self.residuals_m,
            "rmse_m": self.rmse_m,
            "max_abs_residual_m": self.max_abs_residual_m,
            "xy_span_m": self.xy_span_m,
            "design_condition": self.design_condition,
            "quality_level": self.quality_level,
            "warnings": self.warnings,
        }


def lrf_point_from_geodetic(
    *,
    id: str,
    lat: float,
    lon: float,
    abs_alt_m: float,
    ltp: LocalTangentPlane,
    source_image_id: str | None = None,
    distance_residual_m: float | None = None,
    weight: float = 1.0,
    metadata: dict[str, Any] | None = None,
) -> LrfPoint:
    """Convert one DJI LRFTargetLat/Lon/AbsAlt record to an LrfPoint."""

    enu = ltp.geodetic_to_enu(lat, lon, abs_alt_m)
    return LrfPoint(
        id=id,
        enu=enu,
        source_image_id=source_image_id,
        distance_residual_m=distance_residual_m,
        weight=weight,
        metadata=metadata,
    )


def fit_lrf_height_plane(
    points: Sequence[LrfPoint],
    *,
    max_abs_residual_m: float | None = None,
    max_iterations: int = 3,
) -> HeightPlane:
    """Fit a local height plane from multiple LRF hit points.

    When max_abs_residual_m is provided, the function performs simple residual
    trimming.  This is useful because some LRF points may hit roofs, trees, or
    vertical surfaces while the intended model is a nearby ground/bridge plane.
    """

    candidates = list(points)
    if len(candidates) < 3:
        raise ValueError("at least three LRF points are required to fit a plane")

    inliers = candidates
    rejected: list[LrfPoint] = []
    for _ in range(max(1, int(max_iterations))):
        plane_values, residuals = _fit_height_plane_once(inliers)
        if max_abs_residual_m is None:
            break
        next_inliers = [
            point
            for point, residual in zip(inliers, residuals)
            if abs(residual) <= max_abs_residual_m
        ]
        if len(next_inliers) < 3 or len(next_inliers) == len(inliers):
            break
        rejected.extend(point for point in inliers if point not in next_inliers)
        inliers = next_inliers

    coefficients, residuals = _fit_height_plane_once(inliers)
    residual_list = [float(v) for v in residuals]
    xy_span = _xy_span_m(inliers)
    condition = _height_plane_design_condition(inliers)
    normal_z = float(abs(normalize(np.array([-coefficients[0], -coefficients[1], 1.0]))[2]))
    rmse_value = _rmse_1d(residual_list)
    max_abs_residual = float(max(abs(v) for v in residual_list)) if residual_list else math.nan
    quality, warnings = estimate_height_plane_quality(
        point_count=len(inliers),
        xy_span_m=xy_span,
        design_condition=condition,
        normal_z=normal_z,
        rmse_m=rmse_value,
        max_abs_residual_m=max_abs_residual,
    )
    return HeightPlane(
        a=float(coefficients[0]),
        b=float(coefficients[1]),
        c=float(coefficients[2]),
        point_count=len(candidates),
        inlier_ids=[point.id for point in inliers],
        rejected_ids=[point.id for point in rejected],
        residuals_m=residual_list,
        rmse_m=rmse_value,
        max_abs_residual_m=max_abs_residual,
        xy_span_m=xy_span,
        design_condition=condition,
        quality_level=quality,
        warnings=warnings,
    )


def _fit_height_plane_once(points: Sequence[LrfPoint]) -> tuple[np.ndarray, np.ndarray]:
    if len(points) < 3:
        raise ValueError("at least three points are required")
    xyz = np.asarray([point.enu for point in points], dtype=float)
    design = np.column_stack([xyz[:, 0], xyz[:, 1], np.ones(len(points))])
    z = xyz[:, 2]
    weights = np.asarray([point.weight for point in points], dtype=float)
    w_sqrt = np.sqrt(weights)
    coefficients, *_ = np.linalg.lstsq(design * w_sqrt[:, None], z * w_sqrt, rcond=None)
    residuals = z - design @ coefficients
    return coefficients, residuals


def estimate_height_plane_quality(
    *,
    point_count: int,
    xy_span_m: float,
    design_condition: float,
    normal_z: float,
    rmse_m: float,
    max_abs_residual_m: float,
) -> tuple[str, list[str]]:
    """Return a conservative quality label and warnings for an LRF plane."""

    warnings: list[str] = []
    if point_count < 5:
        warnings.append("few LRF points; plane fit is weakly constrained")
    if xy_span_m < 5.0:
        warnings.append("LRF points are too spatially clustered")
    if not np.isfinite(design_condition) or design_condition > 1e6:
        warnings.append("LRF plane fit is ill-conditioned")
    if normal_z < 0.7:
        warnings.append("fitted plane is steep; points may be mixed surfaces or nearly collinear")
    if rmse_m > 3.0 or max_abs_residual_m > 6.0:
        warnings.append("LRF plane residuals are large; do not use as one local surface")

    if (
        point_count >= 5
        and xy_span_m >= 10.0
        and design_condition <= 1e5
        and normal_z >= 0.9
        and rmse_m <= 1.0
        and max_abs_residual_m <= 2.0
    ):
        return "high", warnings
    if (
        point_count >= 3
        and xy_span_m >= 5.0
        and design_condition <= 1e6
        and normal_z >= 0.7
        and rmse_m <= 3.0
        and max_abs_residual_m <= 6.0
    ):
        return "medium", warnings
    return "low", warnings


def _xy_span_m(points: Sequence[LrfPoint]) -> float:
    xy = np.asarray([point.enu[:2] for point in points], dtype=float)
    if len(xy) < 2:
        return 0.0
    centered = xy - xy.mean(axis=0)
    return float(2.0 * np.max(np.linalg.norm(centered, axis=1)))


def _height_plane_design_condition(points: Sequence[LrfPoint]) -> float:
    xyz = np.asarray([point.enu for point in points], dtype=float)
    design = np.column_stack([xyz[:, 0], xyz[:, 1], np.ones(len(points))])
    return float(np.linalg.cond(design))


def locate_pixel_on_height_plane(
    pixel: Sequence[float],
    intrinsics: CameraIntrinsics,
    pose: CameraPose,
    plane: HeightPlane,
) -> np.ndarray:
    """Locate one pixel by intersecting its ray with a fitted LRF plane."""

    origin, direction = pixel_to_world_ray(pixel, intrinsics, pose)
    return plane.intersect_ray(origin, direction)


def _rmse_1d(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return math.nan
    return float(math.sqrt(np.mean(arr * arr)))
