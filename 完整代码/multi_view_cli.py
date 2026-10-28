"""Standalone CLI for optional multi-photo localization extensions.

This does not replace locate_cli.py or the existing web workflow.  It is a
separate entry point for testing:

* multi_view_intersection: same target clicked in several photos;
* lrf_height_plane: local plane fitted from several DJI LRF hit points.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from image_localization import (
    CameraIntrinsics,
    CameraPose,
    LocalTangentPlane,
    pixel_to_world_ray,
)
from multi_view_localization import (
    LrfPoint,
    fit_lrf_height_plane,
    locate_pixel_on_height_plane,
    ray_observation_from_pixel,
    triangulate_rays,
)
from photo_localizer_app import (
    LRF_DISTANCE_RESIDUAL_ABS_LIMIT_M,
    LRF_DISTANCE_RESIDUAL_REL_LIMIT,
    pose_from_dji_metadata,
    read_photo_metadata,
    resolve_image_path,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run optional multi-photo localization extensions.")
    parser.add_argument("--config", required=True, help="Path to a multi-view extension JSON config.")
    parser.add_argument("--output", help="Optional output JSON path.")
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    config = resolve_config_image_paths(config, config_path.parent)

    result = run_config(config)
    if args.output:
        with Path(args.output).open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))


def resolve_config_image_paths(value: Any, base_directory: Path, key: str = "") -> Any:
    """Interpret image paths in CLI JSON relative to the configuration file."""
    if isinstance(value, dict):
        return {name: resolve_config_image_paths(item, base_directory, name) for name, item in value.items()}
    if isinstance(value, list):
        return [resolve_config_image_paths(item, base_directory, key) for item in value]
    if isinstance(value, str) and key in {"image_path", "image_paths", "images"}:
        path = Path(value).expanduser()
        return str((base_directory / path).resolve()) if not path.is_absolute() else str(path)
    return value


def run_config(config: dict[str, Any]) -> dict[str, Any]:
    method = str(config.get("method", "multi_view_intersection"))
    if method == "multi_view_intersection":
        return run_multi_view_intersection(config)
    if method == "lrf_height_plane":
        return run_lrf_height_plane(config)
    raise ValueError("method must be 'multi_view_intersection' or 'lrf_height_plane'")


def run_multi_view_intersection(config: dict[str, Any]) -> dict[str, Any]:
    observations_config = list(config.get("observations", []))
    if len(observations_config) < 2:
        raise ValueError("multi_view_intersection requires at least two observations")

    metadata_by_path = {
        str(resolve_image_path(str(obs["image_path"]))): read_photo_metadata(resolve_image_path(str(obs["image_path"])))
        for obs in observations_config
    }
    ltp = local_tangent_plane_for_config(config, list(metadata_by_path.values()))
    global_intrinsics = config.get("intrinsics")
    pose_correction = str(config.get("pose_correction", "none"))

    ray_observations = []
    observation_rows = []
    for obs in observations_config:
        image_path = resolve_image_path(str(obs["image_path"]))
        metadata = metadata_by_path[str(image_path)]
        pixel = pixel_from_config(obs)
        intrinsics = intrinsics_for_observation(obs, global_intrinsics, metadata)
        camera_center = camera_center_enu(metadata, ltp)
        pose, pose_diagnostics = pose_for_observation(
            obs,
            metadata,
            center_enu=camera_center,
            ltp=ltp,
            intrinsics=intrinsics,
            pose_correction=pose_correction,
        )
        ray_obs = ray_observation_from_pixel(
            image_id=str(obs.get("id", image_path.name)),
            pixel=pixel,
            intrinsics=intrinsics,
            pose=pose,
            weight=float(obs.get("weight", 1.0)),
            metadata={"image_path": str(image_path), **pose_diagnostics},
        )
        ray_observations.append(ray_obs)
        observation_rows.append(
            {
                "id": ray_obs.image_id,
                "image_path": str(image_path),
                "pixel": [float(pixel[0]), float(pixel[1])],
                "camera_center_enu": camera_center.tolist(),
                "ray_direction_enu": np.asarray(ray_obs.direction_enu, dtype=float).tolist(),
                **pose_diagnostics,
            }
        )

    triangulated = triangulate_rays(ray_observations)
    lat, lon, h = ltp.enu_to_geodetic(triangulated.point_enu)
    return {
        "method": "multi_view_intersection",
        "origin": origin_dict(ltp),
        "result": {
            **triangulated.to_dict(),
            "geodetic": {"lat": lat, "lon": lon, "h": h},
        },
        "observations": observation_rows,
    }


def run_lrf_height_plane(config: dict[str, Any]) -> dict[str, Any]:
    image_paths = [resolve_image_path(str(p)) for p in config.get("image_paths", config.get("images", []))]
    if len(image_paths) < 3:
        raise ValueError("lrf_height_plane requires at least three images")

    metadata_list = [read_photo_metadata(path) for path in image_paths]
    ltp = local_tangent_plane_for_config(config, metadata_list)
    use_distance_gate = bool(config.get("use_distance_gate", True))
    max_residual = config.get("max_abs_plane_residual_m")
    max_residual_value = None if max_residual is None else float(max_residual)

    points: list[LrfPoint] = []
    skipped: list[dict[str, Any]] = []
    for path, metadata in zip(image_paths, metadata_list):
        xmp = metadata.get("xmp_candidates", {})
        point, diagnostics = lrf_point_for_metadata(
            path.name,
            metadata,
            ltp,
            use_distance_gate=use_distance_gate,
        )
        if point is None:
            skipped.append({"image_path": str(path), **diagnostics})
            continue
        points.append(point)

    if len(points) < 3:
        raise ValueError(f"not enough valid LRF points after filtering: {len(points)}")

    plane = fit_lrf_height_plane(
        points,
        max_abs_residual_m=max_residual_value,
        max_iterations=int(config.get("max_plane_fit_iterations", 3)),
    )
    result: dict[str, Any] = {
        "method": "lrf_height_plane",
        "origin": origin_dict(ltp),
        "plane": plane.to_dict(),
        "lrf_points": [
            {
                "id": p.id,
                "source_image_id": p.source_image_id,
                "enu": np.asarray(p.enu, dtype=float).tolist(),
                "distance_residual_m": p.distance_residual_m,
            }
            for p in points
        ],
        "skipped": skipped,
    }

    targets = list(config.get("targets", []))
    if targets:
        global_intrinsics = config.get("intrinsics")
        located = []
        metadata_cache = {str(path): metadata for path, metadata in zip(image_paths, metadata_list)}
        for target in targets:
            image_path = resolve_image_path(str(target["image_path"]))
            metadata = metadata_cache.get(str(image_path)) or read_photo_metadata(image_path)
            pixel = pixel_from_config(target)
            intrinsics = intrinsics_for_observation(target, global_intrinsics, metadata)
            center = camera_center_enu(metadata, ltp)
            pose, pose_diagnostics = pose_for_observation(
                target,
                metadata,
                center_enu=center,
                ltp=ltp,
                intrinsics=intrinsics,
                pose_correction=str(config.get("pose_correction", "none")),
            )
            point_enu = locate_pixel_on_height_plane(pixel, intrinsics, pose, plane)
            lat, lon, h = ltp.enu_to_geodetic(point_enu)
            located.append(
                {
                    "id": target.get("id", ""),
                    "image_path": str(image_path),
                    "pixel": [float(pixel[0]), float(pixel[1])],
                    "enu": point_enu.tolist(),
                    "geodetic": {"lat": lat, "lon": lon, "h": h},
                    **pose_diagnostics,
                }
            )
        result["targets"] = located
    return result


def local_tangent_plane_for_config(
    config: dict[str, Any],
    metadata_list: list[dict[str, Any]],
) -> LocalTangentPlane:
    origin = config.get("origin")
    if origin is not None:
        return LocalTangentPlane(
            origin_lat_deg=float(origin["lat"]),
            origin_lon_deg=float(origin["lon"]),
            origin_h_m=float(origin.get("h", 0.0)),
        )
    for metadata in metadata_list:
        gps = metadata.get("gps") or {}
        if gps.get("lat") is not None and gps.get("lon") is not None:
            return LocalTangentPlane(
                origin_lat_deg=float(gps["lat"]),
                origin_lon_deg=float(gps["lon"]),
                origin_h_m=float(gps.get("h") or 0.0),
            )
    raise ValueError("origin is required when image GPS is unavailable")


def origin_dict(ltp: LocalTangentPlane) -> dict[str, float]:
    return {
        "lat": float(ltp.origin_lat_deg),
        "lon": float(ltp.origin_lon_deg),
        "h": float(ltp.origin_h_m),
    }


def pixel_from_config(data: dict[str, Any]) -> tuple[float, float]:
    pixel = data.get("pixel")
    if pixel is None:
        raise ValueError("each observation/target needs pixel: [u, v]")
    return float(pixel[0]), float(pixel[1])


def intrinsics_for_observation(
    observation: dict[str, Any],
    global_intrinsics: dict[str, Any] | None,
    metadata: dict[str, Any],
) -> CameraIntrinsics:
    data = observation.get("intrinsics") or global_intrinsics
    if data is not None:
        return CameraIntrinsics(
            fx=float(data["fx"]),
            fy=float(data.get("fy", data["fx"])),
            cx=float(data["cx"]),
            cy=float(data["cy"]),
            distortion=data.get("distortion", [0, 0, 0, 0, 0]),
            width=int(data["width"]) if data.get("width") is not None else None,
            height=int(data["height"]) if data.get("height") is not None else None,
        )

    camera = metadata.get("camera", {})
    image = metadata.get("image", {})
    focal = camera.get("calibrated_focal_length_px") or camera.get("fx_estimate_from_35mm")
    cx = camera.get("calibrated_optical_center_x")
    cy = camera.get("calibrated_optical_center_y")
    if focal is None or cx is None or cy is None:
        raise ValueError(f"missing calibrated intrinsics for {metadata.get('name')}")
    return CameraIntrinsics(
        fx=float(focal),
        fy=float(focal),
        cx=float(cx),
        cy=float(cy),
        distortion=[0, 0, 0, 0, 0],
        width=int(image["width"]) if image.get("width") is not None else None,
        height=int(image["height"]) if image.get("height") is not None else None,
    )


def camera_center_enu(metadata: dict[str, Any], ltp: LocalTangentPlane) -> np.ndarray:
    gps = metadata.get("gps") or {}
    if gps.get("lat") is None or gps.get("lon") is None:
        raise ValueError(f"missing camera GPS for {metadata.get('name')}")
    return ltp.geodetic_to_enu(
        float(gps["lat"]),
        float(gps["lon"]),
        float(gps.get("h") or ltp.origin_h_m),
    )


def pose_for_observation(
    observation: dict[str, Any],
    metadata: dict[str, Any],
    *,
    center_enu: np.ndarray,
    ltp: LocalTangentPlane,
    intrinsics: CameraIntrinsics,
    pose_correction: str,
) -> tuple[CameraPose, dict[str, Any]]:
    if "camera_to_enu" in observation:
        return (
            CameraPose(center_enu=center_enu, camera_to_enu=observation["camera_to_enu"]),
            {"pose_source": "config.camera_to_enu"},
        )
    if "pose" in observation:
        pose = parse_pose_config(observation["pose"], center_enu)
        return pose, {"pose_source": "config.pose"}

    xmp = metadata.get("xmp_candidates", {})
    metadata_pose = pose_from_dji_metadata(xmp)
    if metadata_pose is None:
        raise ValueError(f"missing DJI gimbal pose for {metadata.get('name')}")
    pose = CameraPose(center_enu=center_enu, camera_to_enu=metadata_pose.camera_to_enu)
    diagnostics: dict[str, Any] = {"pose_source": "dji_gimbal"}

    if pose_correction == "lrf_if_valid":
        lrf_point, lrf_diagnostics = lrf_point_for_metadata(
            str(metadata.get("name", "")),
            metadata,
            ltp,
            use_distance_gate=True,
        )
        diagnostics.update({f"lrf_{k}": v for k, v in lrf_diagnostics.items()})
        if lrf_point is not None:
            image_up_hint = -np.asarray(pose.camera_to_enu, dtype=float)[:, 1]
            pose = CameraPose.look_at(
                center_enu=center_enu,
                target_enu=lrf_point.enu,
                image_up_hint_enu=image_up_hint,
            )
            diagnostics["pose_source"] = "lrf_look_at"
    elif pose_correction != "none":
        raise ValueError("pose_correction must be 'none' or 'lrf_if_valid'")

    return pose, diagnostics


def parse_pose_config(data: dict[str, Any], center_enu: np.ndarray) -> CameraPose:
    if "camera_to_enu" in data:
        return CameraPose(center_enu=center_enu, camera_to_enu=data["camera_to_enu"])
    pose_type = str(data.get("type", data.get("mode", "nadir")))
    if pose_type == "nadir":
        return CameraPose.from_heading_tilt(
            center_enu=center_enu,
            heading_deg=float(data.get("image_up_azimuth_deg", 0.0)),
            tilt_from_nadir_deg=0.0,
            roll_deg=float(data.get("roll_deg", 0.0)),
        )
    if pose_type == "heading_tilt":
        return CameraPose.from_heading_tilt(
            center_enu=center_enu,
            heading_deg=float(data.get("heading_deg", 0.0)),
            tilt_from_nadir_deg=float(data.get("tilt_from_nadir_deg", 0.0)),
            roll_deg=float(data.get("roll_deg", 0.0)),
        )
    if pose_type == "look_at":
        return CameraPose.look_at(
            center_enu=center_enu,
            target_enu=data["target_enu"],
            image_up_hint_enu=data.get("image_up_hint_enu", [0, 1, 0]),
        )
    raise ValueError("pose type must be nadir, heading_tilt, look_at, or provide camera_to_enu")


def lrf_point_for_metadata(
    image_id: str,
    metadata: dict[str, Any],
    ltp: LocalTangentPlane,
    *,
    use_distance_gate: bool,
) -> tuple[LrfPoint | None, dict[str, Any]]:
    xmp = metadata.get("xmp_candidates", {})
    status = str(xmp.get("lrf_status", "")).lower()
    if status not in {"normal", ""}:
        return None, {"skip_reason": "lrf_status_not_normal", "lrf_status": status}

    lat = xmp.get("lrf_target_lat")
    lon = xmp.get("lrf_target_lon")
    h = xmp.get("lrf_target_abs_alt_m")
    if lat is None or lon is None or h is None:
        return None, {"skip_reason": "missing_lrf_target"}

    target_enu = ltp.geodetic_to_enu(float(lat), float(lon), float(h))
    diagnostics: dict[str, Any] = {
        "target_enu": target_enu.tolist(),
        "target_lat": float(lat),
        "target_lon": float(lon),
        "target_abs_alt_m": float(h),
    }
    distance_residual = None
    lrf_distance = xmp.get("lrf_target_distance_m")
    gps = metadata.get("gps") or {}
    if lrf_distance is not None and gps.get("lat") is not None and gps.get("lon") is not None:
        camera = camera_center_enu(metadata, ltp)
        distance_from_coordinates = float(np.linalg.norm(target_enu - camera))
        distance_from_metadata = float(lrf_distance)
        distance_residual = abs(distance_from_coordinates - distance_from_metadata)
        limit = max(
            LRF_DISTANCE_RESIDUAL_ABS_LIMIT_M,
            distance_from_metadata * LRF_DISTANCE_RESIDUAL_REL_LIMIT,
        )
        diagnostics.update(
            {
                "distance_m": distance_from_metadata,
                "coordinate_distance_m": distance_from_coordinates,
                "distance_residual_m": distance_residual,
                "distance_limit_m": limit,
                "distance_ok": distance_residual <= limit,
            }
        )
        if use_distance_gate and distance_residual > limit:
            diagnostics["skip_reason"] = "lrf_distance_gate_failed"
            return None, diagnostics
    elif use_distance_gate:
        diagnostics["skip_reason"] = "missing_lrf_distance_or_camera_gps"
        return None, diagnostics

    return (
        LrfPoint(
            id=image_id,
            enu=target_enu,
            source_image_id=image_id,
            distance_residual_m=distance_residual,
            metadata={"image_name": metadata.get("name")},
        ),
        diagnostics,
    )


if __name__ == "__main__":
    main()
