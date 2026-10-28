"""Standalone web app for multi-photo target intersection.

This module is intentionally separate from photo_localizer_app.py so the
existing single-photo workflow remains unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template, request, send_file

from multi_view_cli import run_multi_view_intersection
from photo_localizer_app import (
    IMAGE_EXTENSIONS,
    DEFAULT_PHOTO_DIRECTORY,
    read_photo_metadata,
    resolve_image_path,
)


app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0


@app.get("/")
def index() -> str:
    return render_template("multi_view_localizer.html", default_directory=str(DEFAULT_PHOTO_DIRECTORY))


@app.get("/api/photos")
def list_photos() -> Any:
    directory = resolve_directory(request.args.get("directory", ""))
    photos = []
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            stat = path.stat()
            photos.append(
                {
                    "name": path.name,
                    "path": str(path),
                    "size_bytes": stat.st_size,
                }
            )
    return jsonify({"directory": str(directory), "photos": photos})


@app.get("/api/image")
def photo_file() -> Any:
    return send_file(resolve_image_path(request.args.get("path", "")))


@app.get("/api/metadata")
def photo_metadata() -> Any:
    return jsonify(read_photo_metadata(resolve_image_path(request.args.get("path", ""))))


@app.post("/api/multi-view/localize")
def multi_view_localize() -> Any:
    data = request.get_json(force=True)
    observations = data.get("observations", [])
    if len(observations) < 2:
        raise ValueError("至少需要两张照片的同名点")
    result = run_multi_view_intersection(
        {
            "method": "multi_view_intersection",
            "pose_correction": data.get("pose_correction", "none"),
            "origin": data.get("origin"),
            "observations": observations,
        }
    )
    return jsonify(result)


@app.errorhandler(Exception)
def bad_request(error: Exception) -> Any:
    return jsonify({"error": str(error)}), 400


def resolve_directory(raw: str) -> Path:
    path = Path(raw or DEFAULT_PHOTO_DIRECTORY).expanduser().resolve()
    if not path.exists() or not path.is_dir():
        raise ValueError(f"photo directory does not exist: {path}")
    return path


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5051, debug=False)
