"""Make the upstream KISS-IMU `src/` importable (training.*, models.*, data.*).

The algorithm code lives in <repo>/src and is imported with top-level package
names, exactly as scripts/train.sh runs it (`cd src; python train.py`). We add
that directory to sys.path instead of copying or modifying it.

The default path assumes this file is loaded from the source tree; an installed
wheel must set the KISS_IMU_SRC environment variable instead.
"""
import os
import sys
from pathlib import Path

_DEFAULT_SRC = Path(__file__).resolve().parents[4] / "src"  # ros2_ws/src/kiss_imu_ros/kiss_imu_ros/paths.py -> <repo>/src


def ensure_src_on_path() -> str:
    src = os.environ.get("KISS_IMU_SRC", str(_DEFAULT_SRC))
    src = str(Path(src).resolve())
    if src not in sys.path:
        sys.path.insert(0, src)
    return src
