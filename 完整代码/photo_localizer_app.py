"""Local visual tool for clicking a photo and estimating map coordinates."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import math
import re
from typing import Any

from flask import Flask, jsonify, render_template, request, send_file
import numpy as np
from PIL import ExifTags, Image

from image_localization import (
    CameraIntrinsics,
    CameraPose,
    LocalTangentPlane,
    locate_pixel_on_plane,
    pixel_to_world_ray,
)
from multi_view_localization import ray_observation_from_pixel, triangulate_rays


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}
GPS_INFO_TAG = 34853
GPS_TAGS = ExifTags.GPSTAGS
EXIF_TAGS = ExifTags.TAGS
DATA_DIRECTORY = Path(__file__).resolve().parent.parent / "数据集"
DEFAULT_PHOTO_DIRECTORY = DATA_DIRECTORY / "原始照片"
DJI_MD_PATH = DATA_DIRECTORY / "dji.md"
LRF_DISTANCE_RESIDUAL_ABS_LIMIT_M = 1.0
LRF_DISTANCE_RESIDUAL_REL_LIMIT = 0.02
MD_GPS_MATCH_LIMIT_M = 0.5
TIME_MATCH_LIMIT_S = 2.0
XMP_NUMERIC_FIELDS = {
    "relative_altitude_m": "RelativeAltitude",
    "absolute_altitude_m": "AbsoluteAltitude",
    "gimbal_pitch_deg": "GimbalPitchDegree",
    "gimbal_yaw_deg": "GimbalYawDegree",
    "gimbal_roll_deg": "GimbalRollDegree",
    "flight_yaw_deg": "FlightYawDegree",
    "flight_pitch_deg": "FlightPitchDegree",
    "flight_roll_deg": "FlightRollDegree",
    "flight_x_speed_mps": "FlightXSpeed",
    "flight_y_speed_mps": "FlightYSpeed",
    "flight_z_speed_mps": "FlightZSpeed",
    "calibrated_focal_length_px": "CalibratedFocalLength",
    "calibrated_optical_center_x": "CalibratedOpticalCenterX",
    "calibrated_optical_center_y": "CalibratedOpticalCenterY",
    "rtk_flag": "RtkFlag",
    "rtk_std_lon": "RtkStdLon",
    "rtk_std_lat": "RtkStdLat",
    "rtk_std_hgt": "RtkStdHgt",
    "rtk_diff_age_s": "RtkDiffAge",
    "lrf_target_distance_m": "LRFTargetDistance",
    "lrf_target_lon": "LRFTargetLon",
    "lrf_target_lat": "LRFTargetLat",
    "lrf_target_alt_m": "LRFTargetAlt",
    "lrf_target_abs_alt_m": "LRFTargetAbsAlt",
    "lens_position": "LensPosition",
    "lens_infinite_position": "LensInfinitePosition",
    "lens_temperature_c": "LensTemperature",
    "sensor_temperature_c": "SensorTemperature",
    "sensor_fps": "SensorFPS",
    "surveying_mode": "SurveyingMode",
    "white_balance_cct": "WhiteBalanceCCT",
    "nd_value": "NdValue",
}
XMP_TEXT_FIELDS = {
    "gps_status": "GpsStatus",
    "altitude_type": "AltitudeType",
    "image_source": "ImageSource",
    "product_name": "ProductName",
    "drone_model": "DroneModel",
    "camera_serial_number": "CameraSerialNumber",
    "drone_serial_number": "DroneSerialNumber",
    "lrf_status": "LRFStatus",
    "shutter_type": "ShutterType",
    "utc_at_exposure": "UTCAtExposure",
}

app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0


@app.get("/")
def index() -> str:
    return render_template("photo_localizer.html", default_directory=str(DEFAULT_PHOTO_DIRECTORY))


@app.get("/api/photos")
def list_photos() -> Any:
    directory = resolve_directory(request.args.get("directory", str(DEFAULT_PHOTO_DIRECTORY)))
    photos = []
    for path in sorted(directory.iterdir(), key=lambda p: p.name.lower()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            stat = path.stat()
            photos.append(
                {
                    "name": path.name,
                    "path": str(path),
                    "size_bytes": stat.st_size,
                    "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
                }
            )
    return jsonify({"directory": str(directory), "photos": photos})


@app.get("/api/image")
def photo_file() -> Any:
    path = resolve_image_path(request.args["path"])
    return send_file(path, conditional=True)


@app.get("/api/metadata")
def photo_metadata() -> Any:
    path = resolve_image_path(request.args["path"])
    return jsonify(read_photo_metadata(path))


@app.post("/api/localize")
def localize_clicked_pixel() -> Any:
    data = request.get_json(force=True)
    pixel = float_pair(data["pixel"], "pixel")
    intrinsics = parse_intrinsics(data["intrinsics"])
    camera_lat = finite_float(data["camera"]["lat"], "camera.lat")
    camera_lon = finite_float(data["camera"]["lon"], "camera.lon")
    camera_h = finite_float(data["camera"].get("h", 0.0), "camera.h")
    height_agl = finite_float(data["ground"]["camera_height_agl_m"], "ground.camera_height_agl_m")
    if height_agl <= 0.0:
        raise ValueError("camera height above ground must be positive")

    ltp = LocalTangentPlane(camera_lat, camera_lon, camera_h)
    nominal_pose = parse_pose(data["pose"])
    pose, plane_z, diagnostics = corrected_pose_and_plane(
        nominal_pose=nominal_pose,
        intrinsics=intrinsics,
        ltp=ltp,
        fallback_plane_z=-height_agl,
        image_path=data.get("image_path"),
        pose_correction=str(data.get("pose_correction", "none")),
    )
    origin, direction = pixel_to_world_ray(pixel, intrinsics, pose)

    agl_result = localize_height_model(
        name="agl",
        label="AGL ground plane",
        pixel=pixel,
        intrinsics=intrinsics,
        pose=pose,
        ltp=ltp,
        plane_z=-height_agl,
        plane_source="camera AbsoluteAltitude - RelativeAltitude",
    )
    lrf_result = None
    if diagnostics.get("lrf_plane_z_enu") is not None:
        lrf_result = localize_height_model(
            name="lrf",
            label="LRF target altitude",
            pixel=pixel,
            intrinsics=intrinsics,
            pose=pose,
            ltp=ltp,
            plane_z=float(diagnostics["lrf_plane_z_enu"]),
            plane_source="LRFTargetAbsAlt",
        )

    height_results = {"agl": agl_result, "lrf": lrf_result}
    height_comparison = compare_height_results(agl_result, lrf_result)
    if height_comparison:
        diagnostics["height_model_horizontal_diff_m"] = height_comparison["horizontal_m"]
        diagnostics["height_model_vertical_diff_m"] = height_comparison["vertical_m"]

    primary_result = lrf_result if lrf_result and not lrf_result.get("error") and diagnostics.get("lrf_used") else agl_result
    if primary_result.get("error"):
        raise ValueError(str(primary_result["error"]))
    point_enu = np.asarray(primary_result["enu"], dtype=float)
    try:
        raw_point_enu = locate_pixel_on_plane(pixel, intrinsics, nominal_pose, plane_z=-height_agl)
        raw_lat, raw_lon, raw_h = ltp.enu_to_geodetic(raw_point_enu)
        diagnostics["raw_enu"] = to_list(raw_point_enu)
        diagnostics["raw_geodetic"] = {"lat": raw_lat, "lon": raw_lon, "h": raw_h}
        diagnostics["corrected_shift_from_raw_m"] = float(np.linalg.norm((point_enu - raw_point_enu)[:2]))
    except ValueError as error:
        diagnostics["raw_localization_error"] = str(error)
    geodetic = primary_result["geodetic"]
    return jsonify(
        {
            "pixel": [pixel[0], pixel[1]],
            "ray": {
                "origin_enu": to_list(origin),
                "direction_enu": to_list(direction),
            },
            "ground_plane_z_enu": primary_result["ground_plane_z_enu"],
            "enu": to_list(point_enu),
            "geodetic": geodetic,
            "height_results": height_results,
            "height_comparison": height_comparison,
            "diagnostics": diagnostics,
        }
    )


@app.post("/api/multi-view/localize")
def localize_multi_view_target() -> Any:
    data = request.get_json(force=True)
    observations = list(data.get("observations", []))
    if len(observations) < 2:
        raise ValueError("multi-view localization needs at least two observations")

    metadata_rows = []
    for observation in observations:
        path = resolve_image_path(str(observation["image_path"]))
        metadata_rows.append((path, read_photo_metadata(path), observation))

    ltp = multi_view_ltp(data.get("origin"), [metadata for _, metadata, _ in metadata_rows])
    global_intrinsics = data.get("intrinsics")
    pose_correction = str(data.get("pose_correction", "none"))
    rays = []
    rows = []
    for path, metadata, observation in metadata_rows:
        pixel = float_pair(observation["pixel"], "observation.pixel")
        intrinsics = multi_view_intrinsics(metadata, observation.get("intrinsics") or global_intrinsics)
        center = multi_view_camera_center(metadata, ltp)
        pose, diagnostics = multi_view_pose(
            metadata=metadata,
            center_enu=center,
            ltp=ltp,
            pose_correction=pose_correction,
        )
        ray_observation = ray_observation_from_pixel(
            image_id=str(observation.get("id") or path.name),
            pixel=pixel,
            intrinsics=intrinsics,
            pose=pose,
            metadata={"image_path": str(path), **diagnostics},
        )
        single_photo = multi_view_single_photo_localization(
            pixel=pixel,
            intrinsics=intrinsics,
            pose=pose,
            center_enu=center,
            metadata=metadata,
            ltp=ltp,
            pose_diagnostics=diagnostics,
        )
        rays.append(ray_observation)
        rows.append(
            {
                "id": ray_observation.image_id,
                "image_path": str(path),
                "pixel": [pixel[0], pixel[1]],
                "camera_center_enu": to_list(center),
                "ray_direction_enu": to_list(ray_observation.direction_enu),
                "single_photo": single_photo,
                **diagnostics,
            }
        )

    result = triangulate_rays(rays)
    for row in rows:
        single_photo = row.get("single_photo", {})
        height_results = single_photo.get("height_results", {})
        for model_result in height_results.values():
            if model_result and not model_result.get("error") and model_result.get("enu") is not None:
                delta = np.asarray(model_result["enu"], dtype=float) - result.point_enu
                model_result["delta_to_intersection"] = {
                    "horizontal_m": float(np.linalg.norm(delta[:2])),
                    "vertical_m": float(abs(delta[2])),
                    "spatial_m": float(np.linalg.norm(delta)),
                    "enu": to_list(delta),
                }
        primary = single_photo.get("primary")
        if primary and primary.get("delta_to_intersection"):
            single_photo["delta_to_intersection"] = primary["delta_to_intersection"]
    lat, lon, h = ltp.enu_to_geodetic(result.point_enu)
    payload = result.to_dict()
    payload["geodetic"] = {"lat": lat, "lon": lon, "h": h}
    return jsonify(
        {
            "origin": {
                "lat": ltp.origin_lat_deg,
                "lon": ltp.origin_lon_deg,
                "h": ltp.origin_h_m,
            },
            "result": payload,
            "observations": rows,
        }
    )


@app.errorhandler(ValueError)
@app.errorhandler(KeyError)
def bad_request(error: Exception) -> Any:
    return jsonify({"error": str(error)}), 400


def resolve_directory(raw: str) -> Path:
    path = Path(raw).expanduser().resolve()
    if not path.exists() or not path.is_dir():
        raise ValueError(f"photo directory does not exist: {path}")
    return path


def resolve_image_path(raw: str) -> Path:
    path = Path(raw).expanduser().resolve()
    if not path.exists() or not path.is_file():
        raise ValueError(f"image does not exist: {path}")
    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError(f"unsupported image extension: {path.suffix}")
    return path


def multi_view_ltp(origin: dict[str, Any] | None, metadata_rows: list[dict[str, Any]]) -> LocalTangentPlane:
    if origin:
        return LocalTangentPlane(
            origin_lat_deg=finite_float(origin["lat"], "origin.lat"),
            origin_lon_deg=finite_float(origin["lon"], "origin.lon"),
            origin_h_m=finite_float(origin.get("h", 0.0), "origin.h"),
        )
    for metadata in metadata_rows:
        gps = metadata.get("gps") or {}
        if gps.get("lat") is not None and gps.get("lon") is not None:
            return LocalTangentPlane(
                origin_lat_deg=float(gps["lat"]),
                origin_lon_deg=float(gps["lon"]),
                origin_h_m=float(gps.get("h") or 0.0),
            )
    raise ValueError("multi-view origin requires at least one photo GPS")


def multi_view_intrinsics(metadata: dict[str, Any], override: dict[str, Any] | None = None) -> CameraIntrinsics:
    if override is not None and {"fx", "cx"}.issubset(override):
        return parse_intrinsics(override)

    image = metadata.get("image", {})
    camera = metadata.get("camera", {})
    width = int(image.get("width") or 0)
    height = int(image.get("height") or 0)
    focal = camera.get("calibrated_focal_length_px") or camera.get("fx_estimate_from_35mm")
    if focal is None:
        raise ValueError(f"missing focal length for {metadata.get('name')}")
    cx = camera.get("calibrated_optical_center_x")
    cy = camera.get("calibrated_optical_center_y")
    return CameraIntrinsics(
        fx=float(focal),
        fy=float(focal),
        cx=float(cx if cx is not None else width * 0.5),
        cy=float(cy if cy is not None else height * 0.5),
        distortion=override.get("distortion", (0.0, 0.0, 0.0, 0.0, 0.0)) if override else (0.0, 0.0, 0.0, 0.0, 0.0),
        width=width or None,
        height=height or None,
    )


def multi_view_camera_center(metadata: dict[str, Any], ltp: LocalTangentPlane) -> np.ndarray:
    gps = metadata.get("gps") or {}
    if gps.get("lat") is None or gps.get("lon") is None:
        raise ValueError(f"missing GPS for {metadata.get('name')}")
    return ltp.geodetic_to_enu(
        float(gps["lat"]),
        float(gps["lon"]),
        float(gps.get("h") or ltp.origin_h_m),
    )


def multi_view_pose(
    *,
    metadata: dict[str, Any],
    center_enu: np.ndarray,
    ltp: LocalTangentPlane,
    pose_correction: str,
) -> tuple[CameraPose, dict[str, Any]]:
    xmp = metadata.get("xmp_candidates", {})
    metadata_pose = pose_from_dji_metadata(xmp)
    if metadata_pose is None:
        raise ValueError(f"missing DJI gimbal pose for {metadata.get('name')}")
    pose = CameraPose(center_enu=center_enu, camera_to_enu=metadata_pose.camera_to_enu)
    diagnostics: dict[str, Any] = {"pose_source": "dji_gimbal"}
    if pose_correction == "none":
        return pose, diagnostics
    if pose_correction != "lrf_if_valid":
        raise ValueError("pose_correction must be none or lrf_if_valid")

    lrf_target = lrf_target_enu(xmp, ltp)
    lrf_status = str(xmp.get("lrf_status", "")).lower()
    if lrf_target is None or lrf_status not in {"normal", ""}:
        diagnostics["lrf_pose_correction"] = "unavailable"
        return pose, diagnostics
    lrf_ok, lrf_diagnostics = multi_view_lrf_distance_gate(xmp, center_enu, lrf_target)
    diagnostics.update(lrf_diagnostics)
    if not lrf_ok:
        diagnostics["lrf_pose_correction"] = "blocked"
        return pose, diagnostics

    image_up_hint = -np.asarray(pose.camera_to_enu, dtype=float)[:, 1]
    corrected = CameraPose.look_at(
        center_enu=center_enu,
        target_enu=lrf_target,
        image_up_hint_enu=image_up_hint,
    )
    diagnostics["pose_source"] = "lrf_look_at"
    diagnostics["lrf_pose_correction"] = "used"
    return corrected, diagnostics


def multi_view_single_photo_localization(
    *,
    pixel: tuple[float, float],
    intrinsics: CameraIntrinsics,
    pose: CameraPose,
    center_enu: np.ndarray,
    metadata: dict[str, Any],
    ltp: LocalTangentPlane,
    pose_diagnostics: dict[str, Any],
) -> dict[str, Any]:
    xmp = metadata.get("xmp_candidates", {})
    agl_result: dict[str, Any] = {
        "name": "agl",
        "label": "AGL ground plane",
        "plane_source": "camera AbsoluteAltitude - RelativeAltitude",
        "error": "missing RelativeAltitude",
    }
    relative_altitude = xmp.get("relative_altitude_m")
    if relative_altitude is not None:
        height_agl = float(relative_altitude)
        if height_agl > 0.0:
            agl_result = localize_height_model(
                name="agl",
                label="AGL ground plane",
                pixel=pixel,
                intrinsics=intrinsics,
                pose=pose,
                ltp=ltp,
                plane_z=float(center_enu[2] - height_agl),
                plane_source="camera AbsoluteAltitude - RelativeAltitude",
            )
        else:
            agl_result["error"] = "RelativeAltitude must be positive"

    lrf_result = None
    lrf_target = lrf_target_enu(xmp, ltp)
    lrf_status = str(xmp.get("lrf_status", "")).lower()
    if lrf_target is not None and lrf_status in {"normal", ""}:
        lrf_ok, lrf_diagnostics = multi_view_lrf_distance_gate(xmp, center_enu, lrf_target)
        if lrf_ok:
            lrf_result = localize_height_model(
                name="lrf",
                label="LRF target altitude",
                pixel=pixel,
                intrinsics=intrinsics,
                pose=pose,
                ltp=ltp,
                plane_z=float(lrf_target[2]),
                plane_source="LRFTargetAbsAlt",
            )
            lrf_result["diagnostics"] = lrf_diagnostics
        else:
            lrf_result = {
                "name": "lrf",
                "label": "LRF target altitude",
                "plane_source": "LRFTargetAbsAlt",
                "ground_plane_z_enu": float(lrf_target[2]),
                "error": "LRF distance gate failed",
                "diagnostics": lrf_diagnostics,
            }
    elif lrf_target is not None:
        lrf_result = {
            "name": "lrf",
            "label": "LRF target altitude",
            "plane_source": "LRFTargetAbsAlt",
            "ground_plane_z_enu": float(lrf_target[2]),
            "error": f"LRF status {lrf_status or 'unknown'}",
        }

    primary = agl_result
    if (
        lrf_result
        and not lrf_result.get("error")
        and (pose_diagnostics.get("lrf_pose_correction") == "used" or agl_result.get("error"))
    ):
        primary = lrf_result

    return {
        "primary_model": primary.get("name"),
        "primary": primary,
        "height_results": {"agl": agl_result, "lrf": lrf_result},
        "height_comparison": compare_height_results(agl_result, lrf_result),
    }


def multi_view_lrf_distance_gate(
    xmp: dict[str, Any],
    center_enu: np.ndarray,
    lrf_target_enu_value: np.ndarray,
) -> tuple[bool, dict[str, Any]]:
    lrf_distance = xmp.get("lrf_target_distance_m")
    if lrf_distance is None:
        return False, {"lrf_distance_ok": False, "lrf_distance_warning": "missing LRFTargetDistance"}
    distance_from_metadata = float(lrf_distance)
    distance_from_coordinates = float(np.linalg.norm(lrf_target_enu_value - center_enu))
    residual = abs(distance_from_coordinates - distance_from_metadata)
    limit = max(LRF_DISTANCE_RESIDUAL_ABS_LIMIT_M, distance_from_metadata * LRF_DISTANCE_RESIDUAL_REL_LIMIT)
    return residual <= limit, {
        "lrf_distance_m": distance_from_metadata,
        "lrf_coordinate_distance_m": distance_from_coordinates,
        "lrf_distance_residual_m": residual,
        "lrf_distance_limit_m": limit,
        "lrf_distance_ok": residual <= limit,
    }


def parse_intrinsics(data: dict[str, Any]) -> CameraIntrinsics:
    distortion = data.get("distortion", [0.0, 0.0, 0.0, 0.0, 0.0])
    return CameraIntrinsics(
        fx=positive_float(data["fx"], "intrinsics.fx"),
        fy=positive_float(data["fy"], "intrinsics.fy"),
        cx=finite_float(data["cx"], "intrinsics.cx"),
        cy=finite_float(data["cy"], "intrinsics.cy"),
        distortion=[finite_float(v, "intrinsics.distortion") for v in distortion],
    )


def parse_pose(data: dict[str, Any]) -> CameraPose:
    mode = str(data.get("mode", "nadir"))
    center = (0.0, 0.0, 0.0)
    if mode == "nadir":
        return CameraPose.from_heading_tilt(
            center_enu=center,
            heading_deg=finite_float(data.get("image_up_azimuth_deg", 0.0), "pose.image_up_azimuth_deg"),
            tilt_from_nadir_deg=0.0,
            roll_deg=finite_float(data.get("roll_deg", 0.0), "pose.roll_deg"),
        )
    if mode == "heading_tilt":
        return CameraPose.from_heading_tilt(
            center_enu=center,
            heading_deg=finite_float(data.get("heading_deg", 0.0), "pose.heading_deg"),
            tilt_from_nadir_deg=finite_float(data.get("tilt_from_nadir_deg", 0.0), "pose.tilt_from_nadir_deg"),
            roll_deg=finite_float(data.get("roll_deg", 0.0), "pose.roll_deg"),
        )
    raise ValueError("pose.mode must be nadir or heading_tilt")


def localize_height_model(
    *,
    name: str,
    label: str,
    pixel: tuple[float, float],
    intrinsics: CameraIntrinsics,
    pose: CameraPose,
    ltp: LocalTangentPlane,
    plane_z: float,
    plane_source: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "name": name,
        "label": label,
        "plane_source": plane_source,
        "ground_plane_z_enu": plane_z,
    }
    try:
        point_enu = locate_pixel_on_plane(pixel, intrinsics, pose, plane_z=plane_z)
    except ValueError as error:
        result["error"] = str(error)
        return result

    lat, lon, h = ltp.enu_to_geodetic(point_enu)
    result.update(
        {
            "enu": to_list(point_enu),
            "geodetic": {"lat": lat, "lon": lon, "h": h},
        }
    )
    return result


def compare_height_results(agl_result: dict[str, Any], lrf_result: dict[str, Any] | None) -> dict[str, Any] | None:
    if not lrf_result or agl_result.get("error") or lrf_result.get("error"):
        return None
    agl_enu = np.asarray(agl_result["enu"], dtype=float)
    lrf_enu = np.asarray(lrf_result["enu"], dtype=float)
    delta = lrf_enu - agl_enu
    return {
        "horizontal_m": float(np.linalg.norm(delta[:2])),
        "vertical_m": float(abs(delta[2])),
        "spatial_m": float(np.linalg.norm(delta)),
        "lrf_minus_agl_enu": to_list(delta),
    }


def corrected_pose_and_plane(
    *,
    nominal_pose: CameraPose,
    intrinsics: CameraIntrinsics,
    ltp: LocalTangentPlane,
    fallback_plane_z: float,
    image_path: Any,
    pose_correction: str,
) -> tuple[CameraPose, float, dict[str, Any]]:
    diagnostics: dict[str, Any] = {
        "correction_mode": "horizontal_plane",
        "plane_source": "relative_altitude",
        "agl_plane_z_enu": fallback_plane_z,
        "lrf_used": False,
        "lrf_gate_passed": False,
    }
    if pose_correction not in {"none", "lrf_if_valid"}:
        raise ValueError("pose_correction must be none or lrf_if_valid")
    if not image_path:
        diagnostics["quality_level"] = estimate_quality_level(diagnostics)
        return nominal_pose, fallback_plane_z, diagnostics

    try:
        metadata = read_photo_metadata(resolve_image_path(str(image_path)))
    except (OSError, ValueError):
        diagnostics["warning"] = "image metadata unavailable"
        diagnostics["quality_level"] = estimate_quality_level(diagnostics)
        return nominal_pose, fallback_plane_z, diagnostics

    xmp = metadata.get("xmp_candidates", {})
    diagnostics.update(metadata.get("metadata_match", {}))
    lrf_target = lrf_target_enu(xmp, ltp)
    metadata_pose = pose_from_dji_metadata(xmp)
    if metadata_pose is not None:
        diagnostics["pose_source"] = "dji_gimbal_absolute_with_flight_check"
        nominal_pose = metadata_pose
    diagnostics.update(attitude_diagnostics(xmp))
    if lrf_target is None or str(xmp.get("lrf_status", "")).lower() not in {"normal", ""}:
        diagnostics.update(quality_diagnostics(xmp))
        diagnostics["quality_level"] = estimate_quality_level(diagnostics)
        return nominal_pose, fallback_plane_z, diagnostics

    plane_z = float(lrf_target[2])
    diagnostics["lrf_plane_z_enu"] = plane_z
    optical_pixel = (intrinsics.cx, intrinsics.cy)
    diagnostics.update(lrf_diagnostics(xmp, lrf_target, nominal_pose, intrinsics, plane_z, optical_pixel))
    diagnostics["lrf_gate_passed"] = bool(diagnostics.get("lrf_distance_ok", False))
    if pose_correction == "none":
        diagnostics.update(quality_diagnostics(xmp))
        diagnostics["quality_level"] = estimate_quality_level(diagnostics)
        return nominal_pose, fallback_plane_z, diagnostics
    if not diagnostics.get("lrf_distance_ok", False):
        diagnostics["warning"] = "LRF distance residual too large; using uncorrected pose"
        diagnostics["quality_level"] = estimate_quality_level(diagnostics)
        return nominal_pose, fallback_plane_z, diagnostics

    image_up_hint = -np.asarray(nominal_pose.camera_to_enu, dtype=float)[:, 1]
    corrected_pose = CameraPose.look_at(
        center_enu=(0.0, 0.0, 0.0),
        target_enu=lrf_target,
        image_up_hint_enu=image_up_hint,
    )
    diagnostics.update(
        {
            "correction_mode": "lrf_look_at",
            "plane_source": "LRFTargetAbsAlt",
            "lrf_used": True,
            "lrf_gate_passed": True,
        }
    )
    diagnostics["quality_level"] = estimate_quality_level(diagnostics)
    return corrected_pose, plane_z, diagnostics


def lrf_target_enu(xmp: dict[str, Any], ltp: LocalTangentPlane) -> np.ndarray | None:
    lat = xmp.get("lrf_target_lat")
    lon = xmp.get("lrf_target_lon")
    h = xmp.get("lrf_target_abs_alt_m")
    if lat is None or lon is None or h is None:
        return None
    return ltp.geodetic_to_enu(float(lat), float(lon), float(h))


def pose_from_dji_metadata(xmp: dict[str, Any]) -> CameraPose | None:
    yaw = xmp.get("gimbal_yaw_deg")
    pitch = xmp.get("gimbal_pitch_deg")
    if yaw is None or pitch is None:
        return None

    tilt = max(0.0, min(89.9, 90.0 + float(pitch)))
    roll = float(xmp.get("gimbal_roll_deg", 0.0))
    return CameraPose.from_heading_tilt(
        center_enu=(0.0, 0.0, 0.0),
        heading_deg=float(yaw),
        tilt_from_nadir_deg=tilt,
        roll_deg=roll,
    )


def attitude_diagnostics(xmp: dict[str, Any]) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {}
    gimbal_yaw = xmp.get("gimbal_yaw_deg")
    flight_yaw = xmp.get("flight_yaw_deg")
    gimbal_roll = xmp.get("gimbal_roll_deg")
    if gimbal_yaw is not None and flight_yaw is not None:
        yaw_delta = angle_difference_deg(float(gimbal_yaw), float(flight_yaw))
        diagnostics["flight_gimbal_yaw_delta_deg"] = yaw_delta
        roll_abs = abs(normalize_angle_deg(float(gimbal_roll or 0.0)))
        same_heading = abs(yaw_delta) <= 5.0
        reversed_heading = abs(abs(yaw_delta) - 180.0) <= 5.0 and roll_abs >= 150.0
        diagnostics["flight_gimbal_yaw_consistent"] = bool(same_heading or reversed_heading)
        if not diagnostics["flight_gimbal_yaw_consistent"]:
            diagnostics["attitude_warning"] = "flight/gimbal yaw mismatch"

    for key in ("flight_roll_deg", "flight_pitch_deg", "flight_yaw_deg", "gimbal_roll_deg", "gimbal_pitch_deg", "gimbal_yaw_deg"):
        if key in xmp:
            diagnostics[key] = xmp[key]
    return diagnostics


def angle_difference_deg(a: float, b: float) -> float:
    return normalize_angle_deg(a - b)


def normalize_angle_deg(value: float) -> float:
    return (float(value) + 180.0) % 360.0 - 180.0


def lrf_diagnostics(
    xmp: dict[str, Any],
    lrf_target: np.ndarray,
    nominal_pose: CameraPose,
    intrinsics: CameraIntrinsics,
    plane_z: float,
    optical_pixel: tuple[float, float],
) -> dict[str, Any]:
    diagnostics = quality_diagnostics(xmp)
    try:
        nominal_optical_hit = locate_pixel_on_plane(optical_pixel, intrinsics, nominal_pose, plane_z=plane_z)
        miss = lrf_target - nominal_optical_hit
        diagnostics["raw_optical_axis_miss_m"] = float(np.linalg.norm(miss[:2]))
        diagnostics["raw_optical_axis_miss_enu"] = to_list(miss)
    except ValueError as error:
        diagnostics["raw_optical_axis_error"] = str(error)

    lrf_distance = xmp.get("lrf_target_distance_m")
    if lrf_distance is not None:
        distance_from_metadata = float(lrf_distance)
        distance_from_coordinates = float(np.linalg.norm(lrf_target))
        residual = abs(distance_from_coordinates - distance_from_metadata)
        limit = max(LRF_DISTANCE_RESIDUAL_ABS_LIMIT_M, distance_from_metadata * LRF_DISTANCE_RESIDUAL_REL_LIMIT)
        diagnostics["lrf_distance_m"] = distance_from_metadata
        diagnostics["lrf_coordinate_distance_m"] = distance_from_coordinates
        diagnostics["lrf_distance_residual_m"] = residual
        diagnostics["lrf_distance_limit_m"] = limit
        diagnostics["lrf_distance_ok"] = residual <= limit
    else:
        diagnostics["lrf_distance_ok"] = False
        diagnostics["lrf_distance_warning"] = "missing LRFTargetDistance"
    return diagnostics


def quality_diagnostics(xmp: dict[str, Any]) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {}
    std_lon = xmp.get("rtk_std_lon")
    std_lat = xmp.get("rtk_std_lat")
    std_hgt = xmp.get("rtk_std_hgt")
    if std_lon is not None and std_lat is not None:
        diagnostics["rtk_horizontal_std_m"] = math.hypot(float(std_lon), float(std_lat))
    if std_hgt is not None:
        diagnostics["rtk_vertical_std_m"] = float(std_hgt)
    speed_values = [xmp.get("flight_x_speed_mps"), xmp.get("flight_y_speed_mps"), xmp.get("flight_z_speed_mps")]
    if all(v is not None for v in speed_values):
        speed = math.sqrt(sum(float(v) * float(v) for v in speed_values))
        diagnostics["flight_speed_mps"] = speed
        if speed > 1.0:
            diagnostics["motion_warning"] = "high flight speed at exposure"
    if xmp.get("lens_position") is not None and xmp.get("lens_infinite_position") is not None:
        lens_delta = float(xmp["lens_position"]) - float(xmp["lens_infinite_position"])
        diagnostics["lens_focus_delta_steps"] = lens_delta
        if abs(lens_delta) > 2.0:
            diagnostics["lens_warning"] = "lens position is far from infinity"
    for key in (
        "image_source",
        "drone_model",
        "camera_serial_number",
        "drone_serial_number",
        "shutter_type",
        "lens_temperature_c",
        "sensor_temperature_c",
    ):
        if key in xmp:
            diagnostics[key] = xmp[key]
    for key in ("rtk_flag", "rtk_diff_age_s", "gps_status", "lrf_status"):
        if key in xmp:
            diagnostics[key] = xmp[key]
    return diagnostics


def estimate_quality_level(diagnostics: dict[str, Any]) -> str:
    warnings = [
        key
        for key in diagnostics
        if key.endswith("_warning") or key in {"warning", "metadata_match_warning", "attitude_warning"}
    ]
    if diagnostics.get("metadata_match_ok") is False:
        return "low"
    if diagnostics.get("lrf_used") and diagnostics.get("lrf_distance_ok") is True:
        rtk_h = diagnostics.get("rtk_horizontal_std_m")
        if rtk_h is not None and float(rtk_h) <= 0.05 and not warnings:
            return "high"
        return "medium"
    if warnings:
        return "low"
    return "medium"


def read_photo_metadata(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        exif = image.getexif()
        exif_names = named_exif(exif)
        exif_gps = read_gps(exif)
        width, height = image.size

    image_xmp = read_xmp_candidates(path)
    md_metadata = read_dji_md_metadata(path.name)
    md_xmp = md_metadata.get("xmp_candidates", {})
    image_gps = gps_from_xmp(image_xmp) or exif_gps
    orientation = exif_names.get("Orientation")
    exif_capture_time = clean_exif_text(exif_names.get("DateTimeOriginal") or exif_names.get("DateTime"))
    metadata_match = validate_metadata_match(
        image_width=width,
        image_height=height,
        image_gps=image_gps,
        exif_capture_time=exif_capture_time,
        md_metadata=md_metadata,
    )
    use_dji_md = bool(metadata_match.get("metadata_match_ok"))
    xmp = md_xmp if use_dji_md else image_xmp
    gps = merge_gps_with_md_fields(image_gps, md_xmp if use_dji_md else {})
    focal_length_mm = (md_metadata.get("focal_length_mm") if use_dji_md else None) or rational_to_float(
        exif_names.get("FocalLength")
    )
    focal_35mm = (md_metadata.get("focal_length_35mm") if use_dji_md else None) or rational_to_float(
        exif_names.get("FocalLengthIn35mmFilm")
    )
    fx_estimate = estimate_fx_from_35mm(width, focal_35mm)
    return {
        "path": str(path),
        "name": path.name,
        "image": {"width": width, "height": height, "orientation": orientation},
        "gps": gps,
        "camera": {
            "make": (md_metadata.get("make") if use_dji_md else None) or clean_exif_text(exif_names.get("Make")),
            "model": (md_metadata.get("model") if use_dji_md else None) or clean_exif_text(exif_names.get("Model")),
            "focal_length_mm": focal_length_mm,
            "focal_length_35mm": focal_35mm,
            "fx_estimate_from_35mm": fx_estimate,
            "calibrated_focal_length_px": xmp.get("calibrated_focal_length_px"),
            "calibrated_optical_center_x": xmp.get("calibrated_optical_center_x"),
            "calibrated_optical_center_y": xmp.get("calibrated_optical_center_y"),
        },
        "capture_time": (md_metadata.get("capture_time") if use_dji_md else None) or exif_capture_time,
        "xmp_candidates": xmp,
        "metadata_sources": {
            "gps_lat_lon": "JPEG XMP/EXIF",
            "other_fields": "dji.md" if use_dji_md else "JPEG fallback",
            "dji_md_matched": use_dji_md,
        },
        "metadata_match": metadata_match,
    }


def merge_gps_with_md_fields(gps: dict[str, Any] | None, md_xmp: dict[str, Any]) -> dict[str, Any] | None:
    if gps is None:
        return None
    out = dict(gps)
    if "absolute_altitude_m" in md_xmp:
        out["h"] = md_xmp["absolute_altitude_m"]
        out["altitude_source"] = "dji.md AbsoluteAltitude"
    return out


def validate_metadata_match(
    *,
    image_width: int,
    image_height: int,
    image_gps: dict[str, Any] | None,
    exif_capture_time: str,
    md_metadata: dict[str, Any],
) -> dict[str, Any]:
    if not md_metadata:
        return {
            "metadata_match_ok": False,
            "metadata_match_warning": "dji.md section not found",
        }

    diagnostics: dict[str, Any] = {}
    warnings: list[str] = []
    if md_metadata.get("dji_md_filename"):
        diagnostics["metadata_dji_md_filename"] = md_metadata["dji_md_filename"]
    if md_metadata.get("dji_md_lookup_filename"):
        diagnostics["metadata_dji_md_lookup_filename"] = md_metadata["dji_md_lookup_filename"]

    md_width = md_metadata.get("image_width")
    md_height = md_metadata.get("image_height")
    size_ok = md_width == image_width and md_height == image_height
    diagnostics["metadata_image_size_ok"] = size_ok
    if not size_ok:
        warnings.append("image size mismatch")

    md_gps = md_metadata.get("gps_display")
    if image_gps and md_gps and md_gps.get("lat") is not None and md_gps.get("lon") is not None:
        gps_diff = horizontal_distance_m(
            float(image_gps["lat"]),
            float(image_gps["lon"]),
            float(md_gps["lat"]),
            float(md_gps["lon"]),
        )
        diagnostics["metadata_gps_display_diff_m"] = gps_diff
        diagnostics["metadata_gps_ok"] = gps_diff <= MD_GPS_MATCH_LIMIT_M
        if gps_diff > MD_GPS_MATCH_LIMIT_M:
            warnings.append("GPS display mismatch")
    else:
        diagnostics["metadata_gps_ok"] = False
        warnings.append("GPS metadata unavailable")

    time_diff = capture_time_difference_s(exif_capture_time, md_metadata.get("capture_time", ""))
    if time_diff is not None:
        diagnostics["metadata_capture_time_diff_s"] = time_diff
        diagnostics["metadata_capture_time_ok"] = time_diff <= TIME_MATCH_LIMIT_S
        if time_diff > TIME_MATCH_LIMIT_S:
            warnings.append("capture time mismatch")
    else:
        diagnostics["metadata_capture_time_ok"] = False
        warnings.append("capture time unavailable")

    diagnostics["metadata_match_ok"] = not warnings
    if warnings:
        diagnostics["metadata_match_warning"] = "; ".join(warnings)
    return diagnostics


def horizontal_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat_m = (lat1 - lat2) * 111_320.0
    lon_m = (lon1 - lon2) * 111_320.0 * math.cos(math.radians((lat1 + lat2) * 0.5))
    return math.hypot(lat_m, lon_m)


def capture_time_difference_s(a: str, b: str) -> float | None:
    t1 = parse_exif_datetime(a)
    t2 = parse_exif_datetime(b)
    if t1 is None or t2 is None:
        return None
    return abs((t1 - t2).total_seconds())


def parse_exif_datetime(value: str) -> datetime | None:
    if not value:
        return None
    value = value.strip()
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(value[:19], fmt)
        except ValueError:
            continue
    return None


def read_dji_md_metadata(filename: str) -> dict[str, Any]:
    section, matched_filename = read_dji_md_section(filename)
    if section is None:
        return {}

    return {
        "dji_md_lookup_filename": filename,
        "dji_md_filename": matched_filename,
        "image_width": int_from_text(dji_md_value(section, "File", "ImageWidth")),
        "image_height": int_from_text(dji_md_value(section, "File", "ImageHeight")),
        "make": dji_md_value(section, "IFD0", "Make") or dji_md_value(section, "XMP-tiff", "Make"),
        "model": dji_md_value(section, "IFD0", "Model") or dji_md_value(section, "XMP-tiff", "Model"),
        "capture_time": dji_md_value(section, "ExifIFD", "DateTimeOriginal")
        or dji_md_value(section, "XMP-drone-dji", "UTCAtExposure"),
        "gps_display": {
            "lat": coordinate_from_text(dji_md_value(section, "XMP-drone-dji", "GPSLatitude")),
            "lon": coordinate_from_text(dji_md_value(section, "XMP-drone-dji", "GPSLongitude")),
        },
        "focal_length_mm": float_from_text(dji_md_value(section, "ExifIFD", "FocalLength")),
        "focal_length_35mm": float_from_text(dji_md_value(section, "ExifIFD", "FocalLengthIn35mmFormat")),
        "xmp_candidates": read_dji_md_xmp_candidates(section),
    }


def read_dji_md_section(filename: str) -> tuple[str | None, str | None]:
    try:
        text = DJI_MD_PATH.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None, None

    match = None
    matched_filename = None
    for candidate in dji_md_filename_candidates(filename):
        header = re.compile(rf"^========\s+{re.escape(candidate)}\s*$", re.MULTILINE)
        match = header.search(text)
        if match:
            matched_filename = candidate
            break
    if not match or matched_filename is None:
        return None, None
    next_header = re.search(r"^========\s+\S+\s*$", text[match.end() :], re.MULTILINE)
    end = match.end() + next_header.start() if next_header else len(text)
    return text[match.end() : end], matched_filename


def dji_md_filename_candidates(filename: str) -> list[str]:
    candidates = [filename]
    original = extract_original_dji_filename(filename)
    if original and original not in candidates:
        candidates.append(original)
    return candidates


def extract_original_dji_filename(filename: str) -> str | None:
    match = re.search(
        r"(DJI_\d{14}_\d{4}_[A-Za-z0-9]+(?:\.[A-Za-z0-9]+)?)$",
        filename,
        flags=re.IGNORECASE,
    )
    return match.group(1) if match else None


def read_dji_md_xmp_candidates(section: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, key in XMP_NUMERIC_FIELDS.items():
        value = float_from_text(dji_md_value(section, "XMP-drone-dji", key))
        if value is not None:
            out[name] = value
    for name, key in XMP_TEXT_FIELDS.items():
        value = dji_md_value(section, "XMP-drone-dji", key)
        if value:
            out[name] = value
    return out


def dji_md_value(section: str, group: str, key: str) -> str | None:
    pattern = re.compile(rf"^\[{re.escape(group)}\]\s+{re.escape(key)}\s*:\s*(.*?)\s*$", re.MULTILINE)
    values = pattern.findall(section)
    return values[-1].strip() if values else None


def float_from_text(value: str | None) -> float | None:
    if value is None:
        return None
    match = re.search(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", value)
    return float(match.group(0)) if match else None


def int_from_text(value: str | None) -> int | None:
    parsed = float_from_text(value)
    return int(parsed) if parsed is not None else None


def coordinate_from_text(value: str | None) -> float | None:
    if value is None:
        return None
    dms = re.search(
        r"(\d+)\s*deg\s*(\d+)'\s*(\d+(?:\.\d+)?)\"?\s*([NSEW])?",
        value,
    )
    if dms:
        out = float(dms.group(1)) + float(dms.group(2)) / 60.0 + float(dms.group(3)) / 3600.0
        return -out if (dms.group(4) or "").upper() in {"S", "W"} else out
    return float_from_text(value)


def named_exif(exif: Image.Exif) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for source in (exif, nested_exif_ifd(exif)):
        for tag_id, value in source.items():
            out[EXIF_TAGS.get(tag_id, str(tag_id))] = value
    return out


def nested_exif_ifd(exif: Image.Exif) -> dict[Any, Any]:
    try:
        if hasattr(ExifTags, "IFD"):
            return exif.get_ifd(ExifTags.IFD.Exif) or {}
    except Exception:
        return {}
    return {}


def read_gps(exif: Image.Exif) -> dict[str, Any] | None:
    gps_ifd: dict[Any, Any] | None = None
    try:
        if hasattr(ExifTags, "IFD"):
            gps_ifd = exif.get_ifd(ExifTags.IFD.GPSInfo)
    except Exception:
        gps_ifd = None
    if not gps_ifd:
        raw = exif.get(GPS_INFO_TAG)
        gps_ifd = raw if isinstance(raw, dict) else None
    if not gps_ifd:
        return None

    gps = {GPS_TAGS.get(k, str(k)): v for k, v in gps_ifd.items()}
    lat = gps_coordinate(gps.get("GPSLatitude"), gps.get("GPSLatitudeRef"))
    lon = gps_coordinate(gps.get("GPSLongitude"), gps.get("GPSLongitudeRef"))
    if lat is None or lon is None:
        return None
    altitude = rational_to_float(gps.get("GPSAltitude"))
    altitude_ref = gps.get("GPSAltitudeRef", 0)
    if altitude is not None and altitude_ref in (1, b"\x01"):
        altitude *= -1.0
    return {
        "lat": lat,
        "lon": lon,
        "h": altitude,
        "altitude_source": "GPSAltitude" if altitude is not None else None,
    }


def gps_coordinate(values: Any, ref: Any) -> float | None:
    if not values or len(values) < 3:
        return None
    deg = rational_to_float(values[0])
    minute = rational_to_float(values[1])
    second = rational_to_float(values[2])
    if deg is None or minute is None or second is None:
        return None
    value = deg + minute / 60.0 + second / 3600.0
    marker = clean_exif_text(ref).upper()
    return -value if marker in {"S", "W"} else value


def rational_to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        if isinstance(value, tuple) and len(value) == 2:
            return float(value[0]) / float(value[1])
        return float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def clean_exif_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore").strip("\x00 ")
    return str(value).strip("\x00 ")


def estimate_fx_from_35mm(width: int, focal_35mm: float | None) -> float | None:
    if focal_35mm is None or focal_35mm <= 0.0:
        return None
    return float(width) * focal_35mm / 36.0


def read_xmp_candidates(path: Path) -> dict[str, float]:
    patterns = {name: (key, key[0].lower() + key[1:]) for name, key in XMP_NUMERIC_FIELDS.items()}
    try:
        with path.open("rb") as f:
            text = f.read(2_500_000).decode("utf-8", errors="ignore")
    except OSError:
        return {}

    out: dict[str, float] = {}
    for name, keys in patterns.items():
        for key in keys:
            match = re.search(rf"(?:[\w-]+:)?{re.escape(key)}=\"([+-]?[0-9]+(?:\.[0-9]+)?)\"", text)
            if not match:
                match = re.search(
                    rf"{re.escape(key)}\s*:\s*([+-]?[0-9]+(?:\.[0-9]+)?)",
                    text,
                )
            if match:
                out[name] = float(match.group(1))
                break
    for name, keys in {
        "gps_latitude_deg": ("GPSLatitude", "GpsLatitude", "gpsLatitude"),
        "gps_longitude_deg": ("GPSLongitude", "GpsLongitude", "gpsLongitude"),
    }.items():
        value = read_text_coordinate(text, keys)
        if value is not None:
            out[name] = value
    return out


def read_text_coordinate(text: str, keys: tuple[str, ...]) -> float | None:
    for key in keys:
        dms = re.search(
            rf"{re.escape(key)}\s*:\s*([0-9]+)\s*deg\s*([0-9]+)'\s*([0-9]+(?:\.[0-9]+)?)\"\s*([NSEW])",
            text,
        )
        if dms:
            value = float(dms.group(1)) + float(dms.group(2)) / 60.0 + float(dms.group(3)) / 3600.0
            return -value if dms.group(4) in {"S", "W"} else value

        decimal = re.search(
            rf"(?:[\w-]+:)?{re.escape(key)}=\"([+-]?[0-9]+(?:\.[0-9]+)?)\"",
            text,
        )
        if not decimal:
            decimal = re.search(
                rf"{re.escape(key)}\s*:\s*([+-]?[0-9]+(?:\.[0-9]+)?)\b",
                text,
            )
        if decimal:
            return float(decimal.group(1))
    return None


def gps_from_xmp(xmp: dict[str, float]) -> dict[str, Any] | None:
    lat = xmp.get("gps_latitude_deg")
    lon = xmp.get("gps_longitude_deg")
    if lat is None or lon is None:
        return None
    return {
        "lat": lat,
        "lon": lon,
        "h": xmp.get("absolute_altitude_m"),
        "altitude_source": "AbsoluteAltitude" if "absolute_altitude_m" in xmp else None,
    }


def finite_float(value: Any, field: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field} must be numeric") from None
    if not math.isfinite(out):
        raise ValueError(f"{field} must be finite")
    return out


def positive_float(value: Any, field: str) -> float:
    out = finite_float(value, field)
    if out <= 0.0:
        raise ValueError(f"{field} must be positive")
    return out


def float_pair(values: Any, field: str) -> tuple[float, float]:
    if not isinstance(values, (list, tuple)) or len(values) != 2:
        raise ValueError(f"{field} must be [u, v]")
    return finite_float(values[0], f"{field}[0]"), finite_float(values[1], f"{field}[1]")


def to_list(values: Any) -> list[float]:
    return [float(v) for v in values]


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050, debug=False)
