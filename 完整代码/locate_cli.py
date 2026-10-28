"""Command line entry point for image target localization.

Examples:
    python locate_cli.py --config input_planar_gcps_example.json
    python locate_cli.py --config input_planar_gcps_example.json --output results.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from image_localization import (
    CameraIntrinsics,
    CameraPose,
    LocalTangentPlane,
    PlanarHomographyLocalizer,
    RegularGridDEM,
    bbox_to_pixel,
    locate_pixel_on_dem,
    locate_pixel_on_plane,
    rmse,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Locate image pixels/bboxes on map coordinates.")
    parser.add_argument("--config", required=True, help="Path to JSON input config.")
    parser.add_argument("--output", help="Optional output file. Supports .json or .csv.")
    args = parser.parse_args()

    config_path = Path(args.config)
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    result = run_localization(config)

    if args.output:
        write_output(Path(args.output), result)
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))


def run_localization(config: dict[str, Any]) -> dict[str, Any]:
    method = str(config.get("method", "planar_gcps"))
    intrinsics = parse_intrinsics(config["intrinsics"])
    ltp = parse_origin(config.get("origin"))
    targets = config.get("targets", [])
    if not targets:
        raise ValueError("config.targets must contain at least one target")

    if method == "planar_gcps":
        rows, diagnostics = locate_with_planar_gcps(config, intrinsics, ltp, targets)
    elif method == "plane_pose":
        rows, diagnostics = locate_with_plane_pose(config, intrinsics, ltp, targets)
    elif method == "dem_grid":
        rows, diagnostics = locate_with_dem_grid(config, intrinsics, ltp, targets)
    else:
        raise ValueError(f"unknown method: {method}")

    return {
        "method": method,
        "count": len(rows),
        "diagnostics": diagnostics,
        "results": rows,
    }


def parse_intrinsics(data: dict[str, Any]) -> CameraIntrinsics:
    distortion = data.get("distortion", data.get("dist", [0, 0, 0, 0, 0]))
    return CameraIntrinsics(
        fx=float(data["fx"]),
        fy=float(data["fy"]),
        cx=float(data["cx"]),
        cy=float(data["cy"]),
        distortion=distortion,
        width=int(data["width"]) if data.get("width") is not None else None,
        height=int(data["height"]) if data.get("height") is not None else None,
    )


def parse_origin(data: dict[str, Any] | None) -> LocalTangentPlane | None:
    if data is None:
        return None
    return LocalTangentPlane(
        origin_lat_deg=float(data["lat"]),
        origin_lon_deg=float(data["lon"]),
        origin_h_m=float(data.get("h", data.get("height", 0.0))),
    )


def parse_pose(data: dict[str, Any]) -> CameraPose:
    if "camera_to_enu" in data:
        return CameraPose(
            center_enu=data["center_enu"],
            camera_to_enu=data["camera_to_enu"],
        )

    pose_type = str(data.get("type", "nadir"))
    if pose_type == "nadir":
        return CameraPose.from_heading_tilt(
            center_enu=data["center_enu"],
            heading_deg=float(data.get("image_up_azimuth_deg", 0.0)),
            tilt_from_nadir_deg=0.0,
            roll_deg=float(data.get("roll_deg", 0.0)),
        )
    if pose_type == "heading_tilt":
        return CameraPose.from_heading_tilt(
            center_enu=data["center_enu"],
            heading_deg=float(data.get("heading_deg", 0.0)),
            tilt_from_nadir_deg=float(data.get("tilt_from_nadir_deg", 0.0)),
            roll_deg=float(data.get("roll_deg", 0.0)),
        )
    if pose_type == "look_at":
        return CameraPose.look_at(
            center_enu=data["center_enu"],
            target_enu=data["target_enu"],
            image_up_hint_enu=data.get("image_up_hint_enu", [0.0, 1.0, 0.0]),
        )

    raise ValueError("pose.type must be 'nadir', 'heading_tilt', 'look_at', or provide camera_to_enu")


def gcp_ground_xy(gcp: dict[str, Any], ltp: LocalTangentPlane | None) -> np.ndarray:
    if "enu" in gcp:
        enu = np.asarray(gcp["enu"], dtype=float)
        if enu.shape[0] == 2:
            return enu
        return enu[:2]
    if "lla" in gcp or "geodetic" in gcp:
        if ltp is None:
            raise ValueError("origin is required when GCP uses lla/geodetic coordinates")
        lla = gcp.get("lla", gcp.get("geodetic"))
        return ltp.geodetic_to_enu(float(lla[0]), float(lla[1]), float(lla[2] if len(lla) > 2 else 0.0))[:2]
    raise ValueError("each GCP needs either enu or lla/geodetic")


def locate_with_planar_gcps(
    config: dict[str, Any],
    intrinsics: CameraIntrinsics,
    ltp: LocalTangentPlane | None,
    targets: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    gcps = config.get("gcps", [])
    if len(gcps) < 4:
        raise ValueError("planar_gcps method requires at least 4 GCPs")

    image_points = np.asarray([g["image"] for g in gcps], dtype=float)
    ground_xy = np.asarray([gcp_ground_xy(g, ltp) for g in gcps], dtype=float)
    plane_z = float(config.get("plane_z", 0.0))
    undistort = bool(config.get("undistort", True))

    localizer = PlanarHomographyLocalizer.from_gcps(
        image_points,
        ground_xy,
        intrinsics=intrinsics,
        plane_z=plane_z,
        undistort=undistort,
    )

    fit_residuals = []
    for gcp, expected_xy in zip(gcps, ground_xy):
        predicted = localizer.locate_pixel(gcp["image"])[:2]
        fit_residuals.append(predicted - expected_xy)

    checkpoint_diagnostics = evaluate_planar_checkpoints(config.get("checkpoints", []), localizer, ltp)
    rows = [result_row(t, localizer.locate_pixel(target_pixel(t)), ltp) for t in targets]
    diagnostics = {
        "gcp_count": len(gcps),
        "gcp_fit_rmse_m": rmse(fit_residuals),
        "plane_z": plane_z,
        "undistort": undistort,
        **checkpoint_diagnostics,
    }
    return rows, diagnostics


def locate_with_plane_pose(
    config: dict[str, Any],
    intrinsics: CameraIntrinsics,
    ltp: LocalTangentPlane | None,
    targets: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pose = parse_pose(config["pose"])
    plane_z = float(config.get("plane_z", 0.0))
    rows = [
        result_row(t, locate_pixel_on_plane(target_pixel(t), intrinsics, pose, plane_z=plane_z), ltp)
        for t in targets
    ]
    diagnostics = {
        "plane_z": plane_z,
        "pose_center_enu": np.asarray(pose.center_enu, dtype=float).tolist(),
    }
    return rows, diagnostics


def locate_with_dem_grid(
    config: dict[str, Any],
    intrinsics: CameraIntrinsics,
    ltp: LocalTangentPlane | None,
    targets: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pose = parse_pose(config["pose"])
    dem_data = config["dem"]
    dem = RegularGridDEM(
        x0=float(dem_data["x0"]),
        y0=float(dem_data["y0"]),
        dx=float(dem_data["dx"]),
        dy=float(dem_data["dy"]),
        elevation=dem_data["elevation"],
        nodata=dem_data.get("nodata"),
    )
    rows = [
        result_row(t, locate_pixel_on_dem(target_pixel(t), intrinsics, pose, dem.height), ltp)
        for t in targets
    ]
    diagnostics = {
        "pose_center_enu": np.asarray(pose.center_enu, dtype=float).tolist(),
        "dem_shape": list(np.asarray(dem.elevation).shape),
    }
    return rows, diagnostics


def target_pixel(target: dict[str, Any]) -> tuple[float, float]:
    if "pixel" in target:
        u, v = target["pixel"]
        return float(u), float(v)
    if "bbox" in target:
        return bbox_to_pixel(target["bbox"], target.get("anchor", "bottom_center"))
    raise ValueError("each target needs either pixel or bbox")


def evaluate_planar_checkpoints(
    checkpoints: list[dict[str, Any]],
    localizer: PlanarHomographyLocalizer,
    ltp: LocalTangentPlane | None,
) -> dict[str, Any]:
    if not checkpoints:
        return {
            "checkpoint_count": 0,
            "checkpoint_rmse_m": None,
            "checkpoints": [],
        }

    residuals = []
    rows = []
    for cp in checkpoints:
        expected_xy = gcp_ground_xy(cp, ltp)
        predicted = localizer.locate_pixel(cp["image"])[:2]
        residual = predicted - expected_xy
        residuals.append(residual)
        rows.append(
            {
                "id": cp.get("id", ""),
                "image": cp["image"],
                "predicted_xy": [float(predicted[0]), float(predicted[1])],
                "expected_xy": [float(expected_xy[0]), float(expected_xy[1])],
                "residual_xy_m": [float(residual[0]), float(residual[1])],
                "horizontal_error_m": float(np.linalg.norm(residual)),
            }
        )

    return {
        "checkpoint_count": len(checkpoints),
        "checkpoint_rmse_m": rmse(residuals),
        "checkpoints": rows,
    }


def result_row(
    target: dict[str, Any],
    point_enu: np.ndarray,
    ltp: LocalTangentPlane | None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": target.get("id", ""),
        "label": target.get("label", ""),
        "anchor": target.get("anchor", ""),
        "pixel": list(target_pixel(target)),
        "enu": [float(point_enu[0]), float(point_enu[1]), float(point_enu[2])],
    }
    if "bbox" in target:
        row["bbox"] = target["bbox"]
    if ltp is not None:
        lat, lon, h = ltp.enu_to_geodetic(point_enu)
        row["geodetic"] = {"lat": lat, "lon": lon, "h": h}
    return row


def write_output(path: Path, result: dict[str, Any]) -> None:
    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        return
    if suffix == ".csv":
        write_csv(path, result["results"])
        return
    raise ValueError("output file must end with .json or .csv")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "id",
        "label",
        "anchor",
        "pixel_u",
        "pixel_v",
        "enu_x",
        "enu_y",
        "enu_z",
        "lat",
        "lon",
        "h",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            geo = row.get("geodetic", {})
            pixel = row["pixel"]
            enu = row["enu"]
            writer.writerow(
                {
                    "id": row.get("id", ""),
                    "label": row.get("label", ""),
                    "anchor": row.get("anchor", ""),
                    "pixel_u": pixel[0],
                    "pixel_v": pixel[1],
                    "enu_x": enu[0],
                    "enu_y": enu[1],
                    "enu_z": enu[2],
                    "lat": geo.get("lat", ""),
                    "lon": geo.get("lon", ""),
                    "h": geo.get("h", ""),
                }
            )


if __name__ == "__main__":
    main()
