"""Check the public export without original photographs or private metadata."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "完整代码"
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(ROOT / "examples"))


def main() -> None:
    for path in CODE.glob("*.py"):
        compile(path.read_text(encoding="utf-8-sig"), str(path), "exec")

    from synthetic_demo import run_demo
    run_demo()
    print("PASS: synthetic plane, triangulation and LRF geometry against known truth")

    for name, count in (("input_plane_pose_example.json", 2),
                        ("input_planar_gcps_example.json", 3)):
        completed = subprocess.run(
            [sys.executable, "-B", str(CODE / "locate_cli.py"), "--config", str(CODE / name)],
            cwd=ROOT, check=True, capture_output=True, text=True, encoding="utf-8",
        )
        payload = json.loads(completed.stdout)
        if payload["count"] != count:
            raise AssertionError(f"unexpected target count for {name}")
        print(f"PASS: CLI {name}")

    from photo_localizer_app import app as single_app
    from multi_view_app import app as multi_app
    for name, app in (("main", single_app), ("multi", multi_app)):
        client = app.test_client()
        response = client.get("/")
        if response.status_code != 200 or b"<html" not in response.data.lower():
            raise AssertionError(f"{name} page failed")
        for asset in (f"{prefix}.{suffix}" for prefix in (
            "photo_localizer" if name == "main" else "multi_view_localizer",
        ) for suffix in ("css", "js")):
            if client.get(f"/static/{asset}").status_code != 200:
                raise AssertionError(f"missing asset: {asset}")
        # Explicit empty directory in public export is a supported initial state.
        photo_response = client.get("/api/photos", query_string={
            "directory": str(ROOT / "数据集" / "原始照片")
        })
        if photo_response.status_code != 200:
            raise AssertionError(f"{name} photo-list endpoint failed")
        print(f"PASS: {name} HTML, CSS, JavaScript and photo-list endpoint")
    print("All public-export checks passed. Field accuracy was not evaluated.")


if __name__ == "__main__":
    main()
