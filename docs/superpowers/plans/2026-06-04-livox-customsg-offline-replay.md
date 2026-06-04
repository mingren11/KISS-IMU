# Livox CustomMsg Offline Bag Replay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the user validate the KISS-IMU LIO front-end on their real Livox Mid-360 ROS2 bag (`data/mid360_1`, which publishes `livox_ros_driver2/msg/CustomMsg` + `sensor_msgs/Imu`), entirely offline in the conda env — no ROS2, no Livox driver, no GPU required.

**Architecture:** A pure, unit-tested conversion core in the package (`kiss_imu_ros/livox.py`: CustomMsg points → `(xyz, t_norm)`), plus a thin glue script (`tools/replay_livox_bag.py`) that uses the pure-python `rosbags` library to decode the bag (registering the CustomMsg/CustomPoint definitions), buffers IMU, feeds `OnlineEstimator.step(..., scan1_ts=...)`, and saves a top-down trajectory plot + npz.

**Tech Stack:** Python 3.10, numpy, `rosbags` (pure-python, pip-installed in `air-io`), matplotlib, the existing `kiss_imu_ros` package. NO ROS2 / rclpy / Livox driver.

Validated by spike: `rosbags 0.11.3` decodes the bag with registered defs; lidar `offset_time` spans 0→~99.4ms (10Hz sweep); IMU acceleration is in **g** (must scale by 9.80665 → m/s²); ~2341 zero/invalid points per scan.

母 spec / context: `docs/superpowers/specs/2026-06-04-lio-phase2a-real-lidar-correctness-design.md` (§2 deskew; this tool exercises that path on real data).

---

## Conventions

- Repo root `<repo>` = `/home/mingren/work/3DVision/KISS-IMU`. Branch `feat/livox-replay` (already checked out — do NOT touch main).
- Test python: `/home/mingren/anaconda3/envs/air-io/bin/python` (torch, pypose, small_gicp, rosbags, matplotlib). Run package tests from `<repo>/ros2_ws/src/kiss_imu_ros` with `KISS_IMU_SRC=<repo>/src`.
- The user's bag: `<repo>/data/mid360_1/` (gitignored — do NOT commit it). Topics: `/livox/imu` (sensor_msgs/Imu, ~200Hz), `/livox/lidar` (livox_ros_driver2/msg/CustomMsg, ~10Hz).
- Commit after each task; append trailer `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.
- All existing tests (24 passed, 1 skipped) MUST stay green.

## File Structure

| File | Change | Responsibility |
| --- | --- | --- |
| `ros2_ws/src/kiss_imu_ros/kiss_imu_ros/livox.py` | create | pure `custom_points_to_xyz_t(points, drop_zeros=True)` → `(xyz (N,3) float64, t_norm (N,))` |
| `ros2_ws/src/kiss_imu_ros/test/test_livox.py` | create | unit tests for the conversion (fake points) |
| `tools/replay_livox_bag.py` | create | rosbags decode + IMU buffer + acc scaling + OnlineEstimator + plot/npz |
| `ros2_ws/src/kiss_imu_ros/README.md` | modify | "Offline Livox bag replay" usage section |

The conversion core lives in the package (unit-tested with the suite); the bag glue lives in `tools/` (needs the bag + rosbags, not unit-tested — validated by running it on the real bag).

---

## Task 1: Livox CustomMsg → (xyz, t_norm) conversion core

**Files:**
- Create: `ros2_ws/src/kiss_imu_ros/kiss_imu_ros/livox.py`
- Create: `ros2_ws/src/kiss_imu_ros/test/test_livox.py`

`custom_points_to_xyz_t(points, drop_zeros=True)` takes an iterable of point objects each with attributes `x, y, z` (float, meters) and `offset_time` (int/float, ns within the sweep). Returns `(xyz (N,3) float64, t_norm (N,) float64)`: drops exact-(0,0,0) returns (when `drop_zeros`) and any non-finite xyz, then normalizes the surviving `offset_time` to [0,1] via `normalize_per_point_times` (reused from `pointcloud2.py`). Empty result → `((0,3), (0,))` arrays.

- [ ] **Step 1: Write failing test** — `test/test_livox.py`:
```python
import numpy as np
from kiss_imu_ros.livox import custom_points_to_xyz_t


class _P:
    __slots__ = ('x', 'y', 'z', 'offset_time')

    def __init__(self, x, y, z, offset_time):
        self.x, self.y, self.z, self.offset_time = x, y, z, offset_time


def test_converts_and_normalizes_offset_time():
    pts = [_P(1.0, 0.0, 0.0, 0), _P(2.0, 0.0, 0.0, 50_000), _P(3.0, 0.0, 0.0, 100_000)]
    xyz, t = custom_points_to_xyz_t(pts)
    assert xyz.shape == (3, 3) and xyz.dtype == np.float64
    assert np.allclose(xyz[:, 0], [1.0, 2.0, 3.0])
    assert abs(t[0]) < 1e-12 and abs(t[-1] - 1.0) < 1e-12   # offset_time -> [0,1]


def test_drops_zero_returns_and_nonfinite():
    pts = [_P(0.0, 0.0, 0.0, 0),            # invalid zero return -> dropped
           _P(1.0, 2.0, 3.0, 10_000),
           _P(np.nan, 1.0, 1.0, 20_000),    # non-finite -> dropped
           _P(4.0, 5.0, 6.0, 30_000)]
    xyz, t = custom_points_to_xyz_t(pts)
    assert xyz.shape == (2, 3)
    assert t.shape == (2,)
    assert np.allclose(xyz[0], [1.0, 2.0, 3.0]) and np.allclose(xyz[1], [4.0, 5.0, 6.0])


def test_keep_zeros_when_disabled():
    pts = [_P(0.0, 0.0, 0.0, 0), _P(1.0, 1.0, 1.0, 10_000)]
    xyz, t = custom_points_to_xyz_t(pts, drop_zeros=False)
    assert xyz.shape == (2, 3)


def test_empty_input():
    xyz, t = custom_points_to_xyz_t([])
    assert xyz.shape == (0, 3) and t.shape == (0,)
```

- [ ] **Step 2: Run, confirm FAIL**

Run: `cd ros2_ws/src/kiss_imu_ros && KISS_IMU_SRC=/home/mingren/work/3DVision/KISS-IMU/src /home/mingren/anaconda3/envs/air-io/bin/python -m pytest test/test_livox.py -v`
Expected: FAIL (No module named 'kiss_imu_ros.livox').

- [ ] **Step 3: Implement** — `kiss_imu_ros/livox.py`:
```python
"""Livox CustomMsg point conversion (pure, ROS-free, unit-tested).

A Livox `livox_ros_driver2/msg/CustomMsg` carries per-point `offset_time`
(ns within the sweep) which is ideal for deskew. This converts its points to
the (xyz, normalized-time) pair the OnlineEstimator/LOModule consume."""
import numpy as np

from .pointcloud2 import normalize_per_point_times


def custom_points_to_xyz_t(points, drop_zeros: bool = True):
    """Convert Livox CustomMsg points -> (xyz (N,3) float64, t_norm (N,) float64).

    Drops exact (0,0,0) invalid returns (when drop_zeros) and non-finite xyz,
    then normalizes the surviving per-point offset_time to [0,1]."""
    if len(points) == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0,), dtype=np.float64)
    xyz = np.array([(p.x, p.y, p.z) for p in points], dtype=np.float64)
    off = np.array([p.offset_time for p in points], dtype=np.float64)
    mask = np.isfinite(xyz).all(axis=1)
    if drop_zeros:
        mask &= ~(xyz == 0.0).all(axis=1)
    xyz = xyz[mask]
    off = off[mask]
    return xyz, normalize_per_point_times(off)
```

- [ ] **Step 4: Run, confirm PASS**

Run: `cd ros2_ws/src/kiss_imu_ros && KISS_IMU_SRC=/home/mingren/work/3DVision/KISS-IMU/src /home/mingren/anaconda3/envs/air-io/bin/python -m pytest test/test_livox.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add ros2_ws/src/kiss_imu_ros/kiss_imu_ros/livox.py ros2_ws/src/kiss_imu_ros/test/test_livox.py
git commit -m "feat(ros): Livox CustomMsg point conversion (xyz + normalized offset_time)"
```

---

## Task 2: offline bag replay script

**Files:**
- Create: `tools/replay_livox_bag.py`

Thin glue (not unit-tested — needs the bag + rosbags). Decodes the bag, buffers IMU (reuse `ImuBuffer`), scales acc to m/s², runs `OnlineEstimator.step(..., scan1_ts=t_norm)`, saves plot + npz.

- [ ] **Step 1: Implement** — `tools/replay_livox_bag.py`:
```python
"""Offline replay of a Livox (Mid-360) ROS2 bag through the KISS-IMU LIO front-end.

Decodes a rosbag2 .db3 with the pure-python `rosbags` library (no ROS install,
no Livox driver), feeds the OnlineEstimator one scan at a time, and saves a
top-down trajectory plot + npz. Validation is qualitative (these bags have no
ground-truth poses).

Usage:
    python tools/replay_livox_bag.py --bag data/mid360_1 \
        --lo-model small_gicp --device cpu --out results/livox_mid360_1
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore, get_types_from_msg

# make the kiss_imu_ros package importable
_PKG = Path(__file__).resolve().parents[1] / "ros2_ws" / "src" / "kiss_imu_ros"
sys.path.insert(0, str(_PKG))
from kiss_imu_ros.ring_buffer import ImuBuffer            # noqa: E402
from kiss_imu_ros.livox import custom_points_to_xyz_t      # noqa: E402
from kiss_imu_ros.online_estimator import OnlineEstimator, EstimatorConfig  # noqa: E402

_CUSTOM_POINT = """
uint32 offset_time
float32 x
float32 y
float32 z
uint8 reflectivity
uint8 tag
uint8 line
"""
_CUSTOM_MSG = """
std_msgs/Header header
uint64 timebase
uint32 point_num
uint8 lidar_id
uint8[3] rsvd
livox_ros_driver2/CustomPoint[] points
"""


def _typestore():
    ts = get_typestore(Stores.ROS2_HUMBLE)
    types = {}
    types.update(get_types_from_msg(_CUSTOM_POINT, "livox_ros_driver2/msg/CustomPoint"))
    types.update(get_types_from_msg(_CUSTOM_MSG, "livox_ros_driver2/msg/CustomMsg"))
    ts.register(types)
    return ts


def _stamp_sec(header) -> float:
    return header.stamp.sec + header.stamp.nanosec * 1e-9


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--bag', required=True, help='rosbag2 directory (contains .db3 + metadata.yaml)')
    ap.add_argument('--imu-topic', default='/livox/imu')
    ap.add_argument('--lidar-topic', default='/livox/lidar')
    ap.add_argument('--acc-scale', type=float, default=9.80665,
                    help='multiply IMU accel by this (Livox reports g; 9.80665 -> m/s^2). Use 1.0 if already m/s^2.')
    ap.add_argument('--lo-model', default='small_gicp')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--max-frames', type=int, default=0, help='0 = all scans')
    ap.add_argument('--out', default='results/livox_replay')
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    est = OnlineEstimator(EstimatorConfig(lo_model=args.lo_model, device=args.device))
    buf = ImuBuffer(maxlen=20000)
    last_t = None
    poses = []
    n = 0

    with AnyReader([Path(args.bag)], default_typestore=_typestore()) as reader:
        conns = [c for c in reader.connections if c.topic in (args.imu_topic, args.lidar_topic)]
        for conn, _ts, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, conn.msgtype)
            if conn.topic == args.imu_topic:
                t = _stamp_sec(msg.header)
                a = msg.linear_acceleration
                g = msg.angular_velocity
                buf.append(t, (a.x * args.acc_scale, a.y * args.acc_scale, a.z * args.acc_scale),
                           (g.x, g.y, g.z))
            elif conn.topic == args.lidar_topic:
                t = _stamp_sec(msg.header)
                t0 = last_t if last_t is not None else (t - 0.1)
                ts_arr, acc, gyro = buf.pop_window(t0, t)
                last_t = t
                xyz, t_norm = custom_points_to_xyz_t(msg.points)
                r = est.step(xyz, acc, gyro, ts_arr, scan1_ts=t_norm)
                poses.append(r.pose)
                n += 1
                if n % 50 == 0:
                    print(f"[replay] {n} scans, pos={r.pose[:3]}, overlap={r.overlap:.3f}, diverged={r.diverged}")
                if args.max_frames and n >= args.max_frames:
                    break

    poses = np.asarray(poses) if poses else np.zeros((0, 7))
    np.savez(out / "trajectory.npz", poses=poses)
    if len(poses):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(poses[:, 0], poses[:, 1], '-', linewidth=1.2)
        ax.scatter([poses[0, 0]], [poses[0, 1]], c='g', s=40, label='start')
        ax.scatter([poses[-1, 0]], [poses[-1, 1]], c='r', s=40, label='end')
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
        ax.set_title(f'Livox replay ({n} scans, {args.lo_model})')
        ax.legend(); plt.tight_layout()
        fig.savefig(out / "trajectory.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
    print(f"[replay] done: {n} scans -> {out}/trajectory.png + trajectory.npz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Byte-compile**

Run: `/home/mingren/anaconda3/envs/air-io/bin/python -m py_compile tools/replay_livox_bag.py && echo OK`
Expected: `OK`.

- [ ] **Step 3: Smoke-run on the real bag (short, first 30 scans, CPU)**

Run:
```bash
cd /home/mingren/work/3DVision/KISS-IMU && KISS_IMU_SRC=$PWD/src /home/mingren/anaconda3/envs/air-io/bin/python tools/replay_livox_bag.py --bag data/mid360_1 --lo-model small_gicp --device cpu --max-frames 30 --out results/livox_mid360_1
```
Expected: prints `[replay] ... scans ...` progress and `[replay] done: 30 scans -> results/livox_mid360_1/trajectory.png + trajectory.npz`, exit 0. The produced poses must be finite (the script prints pos each 50; for 30 frames check the final "done" line appears without traceback).
- If `small_gicp` errors on the real (dense ~17k-point) scans, that is a genuine finding — report it; do NOT silently change behavior. (Backends downsample internally, so it should cope.)
- If it runs but `diverged=True` appears often, note it (acc-scale or extrinsic issue) but the task still passes if it completes without exception and writes outputs.

- [ ] **Step 4: Commit** (do NOT commit `results/` or `data/`)

```bash
git add tools/replay_livox_bag.py
git status --short   # confirm only tools/replay_livox_bag.py staged
git commit -m "feat(tools): offline Livox CustomMsg bag replay through the LIO front-end"
```

---

## Task 3: README + full suite

**Files:**
- Modify: `ros2_ws/src/kiss_imu_ros/README.md`

- [ ] **Step 1: Add a section** to `README.md` after the "Bag replay" section (or near it):
```markdown
## Offline Livox (Mid-360) bag replay — no ROS needed
A Livox Mid-360 typically publishes `livox_ros_driver2/msg/CustomMsg` (not PointCloud2),
which the live node does not subscribe to. To validate the front-end on such a bag
entirely offline (no ROS2, no Livox driver):
```bash
pip install rosbags
python tools/replay_livox_bag.py --bag data/mid360_1 --lo-model small_gicp --device cpu --out results/livox_mid360_1
```
Notes:
- The Mid-360 IMU reports acceleration in **g**; the tool scales by `9.80665` to m/s² by default (`--acc-scale 1.0` if your IMU is already m/s²).
- Per-point `offset_time` is used for deskew (normalized to [0,1]).
- These bags have no ground-truth poses, so validation is qualitative (inspect `trajectory.png`).
- For real-time on-robot use of CustomMsg, a node-side adapter (vendored Livox msgs + `input_mode`) is a separate follow-up.
```

- [ ] **Step 2: Run the FULL package suite (regenerate synth if replay test skips)**

Run:
```bash
/home/mingren/anaconda3/envs/air-io/bin/python /home/mingren/work/3DVision/KISS-IMU/tools/gen_synth_dataset.py --out /home/mingren/work/3DVision/KISS-IMU/data/synth/Synth01
cd ros2_ws/src/kiss_imu_ros && KISS_IMU_SRC=/home/mingren/work/3DVision/KISS-IMU/src /home/mingren/anaconda3/envs/air-io/bin/python -m pytest test/ -v
```
Expected: all pass + 1 skipped (node_config). New: livox 4 tests added → total 28 passed, 1 skipped. Report the summary line. If ANY test FAILS, STOP and report BLOCKED.

- [ ] **Step 3: Commit**

```bash
git add ros2_ws/src/kiss_imu_ros/README.md
git commit -m "docs(ros): document offline Livox bag replay tool"
```

---

## Self-Review (plan vs goal)

- **Goal coverage:** decode user's CustomMsg bag offline → Task 2 (rosbags + registered defs, validated by spike); deskew on real data → Task 1 conversion uses `normalize_per_point_times`, threaded via `step(scan1_ts=)` (Phase 2a); acc-in-g gotcha → Task 2 `--acc-scale` default 9.80665; qualitative validation (no GT) → trajectory plot.
- **Placeholder scan:** none — full code in every step.
- **Type consistency:** `custom_points_to_xyz_t(points, drop_zeros=True) -> (xyz, t_norm)` (Task 1) called in Task 2 as `xyz, t_norm = custom_points_to_xyz_t(msg.points)`; `OnlineEstimator(EstimatorConfig(...))` + `step(xyz, acc, gyro, ts, scan1_ts=t_norm)` match the merged Phase 2a API; `ImuBuffer.append/pop_window` match Phase 1.
- **No commit of data/results:** Task 2/3 explicitly stage only source files; `data/` and `results/` are gitignored.
- **Backward compat:** purely additive (new module + new tool + README); existing tests unaffected (verified in Task 3 full run).
- **Dependency:** `rosbags` is a tool-only dep (already pip-installed in air-io); not added to package runtime deps.
