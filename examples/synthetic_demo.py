"""Deterministic geometry demo. All observations and targets are synthetic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "完整代码"))

import numpy as np

from image_localization import CameraIntrinsics, CameraPose, locate_pixel_on_plane
from multi_view_localization import (
    LrfPoint,
    fit_lrf_height_plane,
    locate_pixel_on_height_plane,
    ray_observation_from_pixel,
    triangulate_rays,
)


def project(point: np.ndarray, pose: CameraPose, intrinsics: CameraIntrinsics) -> np.ndarray:
    """Generate a pixel from known geometry (zero distortion in this demo)."""
    camera_point = pose.camera_to_enu.T @ (point - pose.center_enu)
    if camera_point[2] <= 0:
        raise ValueError("synthetic point must be in front of the camera")
    return np.array([
        intrinsics.fx * camera_point[0] / camera_point[2] + intrinsics.cx,
        intrinsics.fy * camera_point[1] / camera_point[2] + intrinsics.cy,
    ])


def run_demo() -> dict:
    intrinsics = CameraIntrinsics(fx=3000, fy=3000, cx=2000, cy=1500,
                                 width=4000, height=3000)
    truth = np.array([6.0, 4.0, 0.0])
    poses = [CameraPose.nadir(center) for center in (
        [-20.0, 0.0, 100.0], [20.0, 0.0, 100.0], [0.0, 25.0, 110.0]
    )]
    pixels = [project(truth, pose, intrinsics) for pose in poses]
    single = locate_pixel_on_plane(pixels[0], intrinsics, poses[0], plane_z=0.0)

    def triangulate(pixel_noise: list[np.ndarray]):
        return triangulate_rays([
            ray_observation_from_pixel(image_id=f"virtual_{i}", pixel=pixel + noise,
                                       intrinsics=intrinsics, pose=pose)
            for i, (pose, pixel, noise) in enumerate(zip(poses, pixels, pixel_noise))
        ])

    exact = triangulate([np.zeros(2) for _ in poses])
    rng = np.random.default_rng(20261028)
    noisy = triangulate(list(rng.normal(0.0, 0.5, size=(len(poses), 2))))

    coefficients = np.array([0.02, -0.01, 2.0])
    ground_points = [LrfPoint(id=f"synthetic_lrf_{i}",
                             enu=[x, y, coefficients @ [x, y, 1.0]])
                     for i, (x, y) in enumerate((
                         (-20, -20), (20, -20), (20, 20), (-20, 20), (0, 0)
                     ))]
    plane = fit_lrf_height_plane(ground_points)
    slope_truth = np.array([6.0, 4.0, coefficients @ [6.0, 4.0, 1.0]])
    slope_pixel = project(slope_truth, poses[0], intrinsics)
    slope_point = locate_pixel_on_height_plane(slope_pixel, intrinsics, poses[0], plane)

    # Known ground truth checks catch sign, coordinate and intersection regressions.
    np.testing.assert_allclose(single, truth, atol=1e-9, rtol=0)
    np.testing.assert_allclose(exact.point_enu, truth, atol=1e-9, rtol=0)
    np.testing.assert_allclose([plane.a, plane.b, plane.c], coefficients, atol=1e-10, rtol=0)
    np.testing.assert_allclose(slope_point, slope_truth, atol=1e-9, rtol=0)

    return {
        "data_type": "synthetic_only",
        "seed": 20261028,
        "units": "ENU metres; image pixels",
        "ground_truth_enu": truth.tolist(),
        "single_plane": {"point_enu": single.tolist(),
                         "error_3d_m": float(np.linalg.norm(single - truth))},
        "multi_view_exact": {**exact.to_dict(),
                             "error_3d_m": float(np.linalg.norm(exact.point_enu - truth))},
        "multi_view_pixel_noise": {"pixel_noise_std": 0.5, **noisy.to_dict(),
                                   "error_3d_m": float(np.linalg.norm(noisy.point_enu - truth))},
        "lrf_plane": {"expected_coefficients": coefficients.tolist(),
                      "fitted_coefficients": [plane.a, plane.b, plane.c],
                      "fit_rmse_m": plane.rmse_m,
                      "ground_truth_enu": slope_truth.tolist(),
                      "point_enu": slope_point.tolist(),
                      "error_3d_m": float(np.linalg.norm(slope_point - slope_truth))},
        "interpretation": "Numerical demonstration only; not field accuracy evidence.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional JSON output file")
    args = parser.parse_args()
    result = json.dumps(run_demo(), ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        args.output.write_text(result + "\n", encoding="utf-8")
    else:
        print(result)


if __name__ == "__main__":
    main()
