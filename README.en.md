# UAV Visual 3D Localization

[简体中文](README.md) | **English**

A Python toolkit that maps UAV image pixels or bounding boxes to local ENU and geodetic coordinates. It includes single-image plane/DEM intersection, planar GCP homography, multi-view ray triangulation, LRF height-plane fitting, a local browser interface and command-line tools.

Pixels are selected manually or supplied by an external detector. The repository's synthetic demonstration verifies geometry and numerical consistency. Field accuracy requires independent ground truth; the project title does not establish a centimetre-level accuracy claim.

## Setup

Verified locally on Windows x64 with Python 3.13, NumPy 2.1.3, Flask 3.1.0 and Pillow 11.3.0.

```powershell
git clone https://github.com/xmuzr/uav-visual-3d-localization.git
cd uav-visual-3d-localization
python -m venv .venv
.venv\Scripts\Activate.ps1
python -X utf8 -m pip install -r requirements.txt
python -B 完整代码/run.py
```

The main UI opens at <http://127.0.0.1:5050>. Use `--multi` for the independent multi-photo UI at port 5051, `--port` to select another port, or `--no-browser` to suppress browser launch.

The code directory retains its original Chinese name. Place your own images in `数据集/原始照片/`, or select a local directory in the interface. Optional supplementary DJI metadata belongs in `数据集/dji.md`. Original photos, survey metadata, acceptance documents and offline dependency wheels are excluded from this public repository.

## Reproducible examples

These commands require no photos or private dataset:

```powershell
python -B examples/synthetic_demo.py
python -B 完整代码/locate_cli.py --config 完整代码/input_plane_pose_example.json
python -B 完整代码/locate_cli.py --config 完整代码/input_planar_gcps_example.json
python -B scripts/check_repository.py
```

The synthetic demo reconstructs known targets from three virtual cameras, fits a known sloped plane, and includes a seeded pixel-noise example. See [sample output](examples/synthetic_result.json) and [validation scope](docs/VALIDATION.md).

The two multi-view/LRF JSON templates reference local capture filenames. Supply your own images and metadata and replace the placeholder target pixels before using those templates.

## Implementation map

| File under `完整代码/` | Purpose |
| --- | --- |
| `image_localization.py` | Camera model, distortion, plane/DEM intersection, GCP homography and coordinate transforms |
| `multi_view_localization.py` | Weighted ray intersection, LRF plane fitting and diagnostics |
| `locate_cli.py` | Single-image JSON/CSV interface |
| `multi_view_cli.py` | Photo-backed multi-view and LRF JSON interface |
| `photo_localizer_app.py`, `multi_view_app.py` | Local Flask interfaces and metadata handling |
| `templates/`, `static/` | Browser UI resources |

Additional documentation: [architecture](docs/ARCHITECTURE.md), [data preparation](docs/DATA.md), [validation](docs/VALIDATION.md). These detailed documents are in Chinese.

## Assumptions

ENU uses east, north and up in metres. Camera axes are right, down and forward; `camera_to_enu` maps camera-frame vectors to ENU. Multi-view observations must refer to the same stationary point. Ground-plane assumptions, altitude references, camera calibration, pose errors and weak baselines all affect field accuracy. Ray residuals and plane-fit errors measure internal agreement and must not be substituted for independent positioning errors.

Documentation organization was informed by [AeroMind](https://github.com/Gypsophilazd/AeroMind) and [m400-spatial-perception](https://github.com/AcrazyQ/m400-spatial-perception), whose access is controlled by its owner. No reference-project source, SDK or private assets were added during this preparation.

No open-source license is currently specified. Dependencies retain their respective licenses.
