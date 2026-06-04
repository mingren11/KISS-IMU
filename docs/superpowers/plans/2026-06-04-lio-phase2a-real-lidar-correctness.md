# LIO Phase 2a (real-LiDAR correctness + real-time safety) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the kiss_imu_ros LIO front-end correct on real spinning/non-repetitive LiDAR (enable kiss_icp deskew via per-point timestamps) and bound its latency (drop backlogged scans), without touching the estimator's algorithm.

**Architecture:** Add per-point timestamp extraction + [0,1] normalization to the PointCloud2 path (configurable `time_field`, works for Velodyne `time` and Livox Mid-360 offset times), thread it through `OnlineEstimator.step(..., scan1_ts=...)` into `LOModule`'s kiss_icp deskew; add a ROS-free `FrameGate` (non-blocking try-lock) so the node drops scans that arrive while a `step()` is still running; wire optional node-level voxel downsampling (default OFF to avoid double-voxelization).

**Tech Stack:** Python 3.10, ROS2 rclpy, numpy, pytest. (kiss_icp deskew exercised on the real robot; CPU tests use small_gicp / pure-numpy.)

Spec: `docs/superpowers/specs/2026-06-04-lio-phase2a-real-lidar-correctness-design.md`.

---

## Conventions

- Repo root `<repo>` = `/home/mingren/work/3DVision/KISS-IMU`. Package: `<repo>/ros2_ws/src/kiss_imu_ros`.
- Test python: `/home/mingren/anaconda3/envs/air-io/bin/python` (torch, pypose, numpy, small_gicp; NO rclpy). Run tests from `<repo>/ros2_ws/src/kiss_imu_ros` with `KISS_IMU_SRC=<repo>/src`.
- Phase 1 is merged. All existing tests (14 passed, 1 skipped) MUST stay green — backward compatibility is a hard requirement.
- Commit after each task; append trailer `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

## File Structure

| File | Change | Responsibility |
| --- | --- | --- |
| `kiss_imu_ros/pointcloud2.py` | modify | add `normalize_per_point_times`, `voxel_downsample_indexed`; `xyz_from_pointcloud2` returns `(xyz, t_norm_or_None)` with `time_field` |
| `kiss_imu_ros/frame_gate.py` | create | `FrameGate` — non-blocking try-lock + dropped counter |
| `kiss_imu_ros/online_estimator.py` | modify | `step(..., scan1_ts=None)` threaded into `build_window_sample` |
| `kiss_imu_ros/lio_node.py` | modify | `time_field`/`max_step_ms` params, FrameGate, latency log, optional aligned voxel downsample |
| `config/lio.yaml` | modify | add `time_field`, `max_step_ms`; comment voxel_size interaction |
| `README.md` | modify | document deskew config, frame-drop, voxel interaction; move 2a items out of deferred |
| `test/test_pointcloud2.py` | modify | tests for normalize + indexed downsample |
| `test/test_frame_gate.py` | create | FrameGate tests |
| `test/test_online_estimator.py` | modify | step accepts scan1_ts; backward compat |
| `test/test_imu_corrector.py` | modify | build_window_sample threads scan1_ts |

---

## Task 1: per-point time normalization + indexed downsample (pointcloud2.py)

**Files:**
- Modify: `ros2_ws/src/kiss_imu_ros/kiss_imu_ros/pointcloud2.py`
- Modify: `ros2_ws/src/kiss_imu_ros/test/test_pointcloud2.py`
- Modify (call-site fix only): `ros2_ws/src/kiss_imu_ros/kiss_imu_ros/lio_node.py`

- [ ] **Step 1: Add failing tests** — append to `test/test_pointcloud2.py`:

```python
from kiss_imu_ros.pointcloud2 import normalize_per_point_times, voxel_downsample_indexed


def test_normalize_velodyne_relative_seconds():
    t = np.linspace(0.0, 0.1, 50)            # velodyne: relative seconds
    out = normalize_per_point_times(t)
    assert out.shape == (50,)
    assert out.min() == 0.0 and abs(out.max() - 1.0) < 1e-12
    assert np.all(np.diff(out) >= 0)


def test_normalize_livox_offset_nanoseconds():
    t = np.arange(0, 100000, 2000, dtype=np.float64)  # livox: large ns offsets
    out = normalize_per_point_times(t)
    assert abs(out[0]) < 1e-12 and abs(out[-1] - 1.0) < 1e-12
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_normalize_degenerate_all_equal_returns_zeros():
    t = np.full(10, 5.0)
    out = normalize_per_point_times(t)
    assert out.shape == (10,) and np.all(out == 0.0)


def test_normalize_empty():
    out = normalize_per_point_times(np.zeros((0,)))
    assert out.shape == (0,)


def test_voxel_downsample_indexed_returns_aligned_indices():
    pts = np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [5.0, 5.0, 5.0]], dtype=np.float64)
    ds, idx = voxel_downsample_indexed(pts, voxel=1.0)
    assert ds.shape[0] == 2
    assert idx.shape[0] == 2
    assert np.allclose(ds, pts[idx])           # indices select exactly the returned points


def test_voxel_downsample_indexed_disabled_returns_all():
    pts = np.random.randn(7, 3)
    ds, idx = voxel_downsample_indexed(pts, voxel=0.0)
    assert ds.shape[0] == 7 and idx.shape[0] == 7
```

- [ ] **Step 2: Run, confirm FAIL**

Run: `cd ros2_ws/src/kiss_imu_ros && KISS_IMU_SRC=/home/mingren/work/3DVision/KISS-IMU/src /home/mingren/anaconda3/envs/air-io/bin/python -m pytest test/test_pointcloud2.py -v`
Expected: FAIL (ImportError: cannot import name 'normalize_per_point_times').

- [ ] **Step 3: Implement** — replace the body of `pointcloud2.py` with:

```python
"""PointCloud2 <-> numpy helpers (pure-numpy parts are unit-tested without ROS)."""
import numpy as np


def xyz_from_arrays(structured) -> np.ndarray:
    """Take x/y/z from a structured array; return (N,3) float64 with NaN/Inf dropped."""
    xyz = np.stack([structured['x'], structured['y'], structured['z']], axis=-1).astype(np.float64)
    mask = np.isfinite(xyz).all(axis=1)
    return xyz[mask]


def normalize_per_point_times(t) -> np.ndarray:
    """Normalize per-point times to [0,1] within the scan (the kiss_icp deskew
    convention). Works for any units (relative seconds, offset ns, absolute):
    (t - min) / (max - min). Degenerate (all equal) or empty -> zeros/empty."""
    t = np.asarray(t, dtype=np.float64)
    if t.size == 0:
        return t
    lo = float(t.min())
    span = float(t.max()) - lo
    if span <= 0.0:
        return np.zeros_like(t)
    return (t - lo) / span


def voxel_downsample_indexed(pts: np.ndarray, voxel: float):
    """Deterministic voxel downsample; return (downsampled_pts, kept_indices) so a
    caller can downsample an aligned per-point array (e.g. timestamps) identically."""
    if pts.size == 0 or voxel <= 0:
        return pts, np.arange(pts.shape[0])
    grid = np.floor(pts / voxel).astype(np.int64)
    _, idx = np.unique(grid, axis=0, return_index=True)
    idx = np.sort(idx)
    return pts[idx], idx


def voxel_downsample(pts: np.ndarray, voxel: float) -> np.ndarray:
    """Deterministic voxel downsample (unique grid cell, sorted) — mirrors upstream."""
    return voxel_downsample_indexed(pts, voxel)[0]


def xyz_from_pointcloud2(msg, field_names=('x', 'y', 'z'), time_field=None):
    """Parse a sensor_msgs/PointCloud2 into (xyz (N,3) float64, t_norm (N,) or None).

    If ``time_field`` is given AND present in the cloud, per-point times are read,
    finite-aligned with the kept xyz, and normalized to [0,1] (kiss_icp deskew
    convention). Otherwise t_norm is None. Imports ROS lazily."""
    from sensor_msgs_py import point_cloud2
    have_time = bool(time_field) and time_field in [f.name for f in msg.fields]
    read_fields = tuple(field_names) + ((time_field,) if have_time else ())
    structured = point_cloud2.read_points(msg, field_names=read_fields, skip_nans=True)
    xyz = np.stack([structured['x'], structured['y'], structured['z']], axis=-1).astype(np.float64)
    mask = np.isfinite(xyz).all(axis=1)
    xyz = xyz[mask]
    if have_time:
        t = np.asarray(structured[time_field], dtype=np.float64)[mask]
        return xyz, normalize_per_point_times(t)
    return xyz, None
```

- [ ] **Step 4: Fix the one call site so the node stays valid** — in `lio_node.py`, line 74 currently:
```python
        scan = xyz_from_pointcloud2(msg)
```
Change to (Task 4 will wire the real time_field; here just keep the node syntactically correct):
```python
        scan, _ = xyz_from_pointcloud2(msg)
```

- [ ] **Step 5: Run tests + byte-compile node**

Run: `cd ros2_ws/src/kiss_imu_ros && KISS_IMU_SRC=/home/mingren/work/3DVision/KISS-IMU/src /home/mingren/anaconda3/envs/air-io/bin/python -m pytest test/test_pointcloud2.py -v`
Expected: PASS (the 2 original + 6 new = 8 passed).
Run: `/home/mingren/anaconda3/envs/air-io/bin/python -m py_compile ros2_ws/src/kiss_imu_ros/kiss_imu_ros/lio_node.py && echo OK`
Expected: `OK`.

- [ ] **Step 6: Commit**

```bash
git add ros2_ws/src/kiss_imu_ros/kiss_imu_ros/pointcloud2.py ros2_ws/src/kiss_imu_ros/test/test_pointcloud2.py ros2_ws/src/kiss_imu_ros/kiss_imu_ros/lio_node.py
git commit -m "feat(ros): per-point time normalization + indexed voxel downsample"
```

---

## Task 2: FrameGate (frame_gate.py)

**Files:**
- Create: `ros2_ws/src/kiss_imu_ros/kiss_imu_ros/frame_gate.py`
- Create: `ros2_ws/src/kiss_imu_ros/test/test_frame_gate.py`

- [ ] **Step 1: Write failing test** — `test/test_frame_gate.py`:

```python
from kiss_imu_ros.frame_gate import FrameGate


def test_enter_then_busy_then_release():
    gate = FrameGate()
    assert gate.try_enter() is True       # first entry succeeds
    assert gate.try_enter() is False      # busy -> dropped
    assert gate.dropped == 1
    assert gate.try_enter() is False      # still busy -> dropped again
    assert gate.dropped == 2
    gate.exit()
    assert gate.try_enter() is True       # free again
    assert gate.dropped == 2              # successful entry does not increment
    gate.exit()


def test_dropped_starts_zero():
    assert FrameGate().dropped == 0
```

- [ ] **Step 2: Run, confirm FAIL**

Run: `cd ros2_ws/src/kiss_imu_ros && /home/mingren/anaconda3/envs/air-io/bin/python -m pytest test/test_frame_gate.py -v`
Expected: FAIL (No module named 'kiss_imu_ros.frame_gate').

- [ ] **Step 3: Implement** — `kiss_imu_ros/frame_gate.py`:

```python
"""Non-blocking gate that drops backlogged work instead of queuing it.

The scan callback calls try_enter() before processing; if a previous step() is
still running (lock held), the new scan is dropped (counter incremented) so
odometry latency does not accumulate under load."""
import threading


class FrameGate:
    def __init__(self):
        self._lock = threading.Lock()
        self.dropped = 0

    def try_enter(self) -> bool:
        if self._lock.acquire(blocking=False):
            return True
        self.dropped += 1
        return False

    def exit(self) -> None:
        self._lock.release()
```

- [ ] **Step 4: Run, confirm PASS**

Run: `cd ros2_ws/src/kiss_imu_ros && /home/mingren/anaconda3/envs/air-io/bin/python -m pytest test/test_frame_gate.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add ros2_ws/src/kiss_imu_ros/kiss_imu_ros/frame_gate.py ros2_ws/src/kiss_imu_ros/test/test_frame_gate.py
git commit -m "feat(ros): FrameGate — non-blocking drop of backlogged scans"
```

---

## Task 3: thread scan1_ts through the estimator

**Files:**
- Modify: `ros2_ws/src/kiss_imu_ros/kiss_imu_ros/online_estimator.py`
- Modify: `ros2_ws/src/kiss_imu_ros/test/test_online_estimator.py`
- Modify: `ros2_ws/src/kiss_imu_ros/test/test_imu_corrector.py`

`build_window_sample(accels, gyros, imu_ts, scan0, scan1, scan1_ts=None)` already accepts `scan1_ts` (Phase 1). This task plumbs it from `OnlineEstimator.step` and adds the missing tests.

- [ ] **Step 1: Add failing tests**

Append to `test/test_imu_corrector.py`:
```python
def test_build_window_sample_threads_scan1_ts():
    accels, gyros, imu_ts, scan0, scan1 = _fake_window()
    t = np.linspace(0.0, 1.0, scan1.shape[0])
    s = build_window_sample(accels, gyros, imu_ts, scan0, scan1, scan1_ts=t)
    assert len(s['scan1_ts']) == 1
    assert s['scan1_ts'][0].shape == (scan1.shape[0],)
    assert torch.allclose(s['scan1_ts'][0].cpu().double(), torch.from_numpy(t))
```

Append to `test/test_online_estimator.py`:
```python
def test_step_accepts_scan1_ts():
    cfg = EstimatorConfig(lo_model='small_gicp', device='cpu')
    est = OnlineEstimator(cfg)
    est.step(_synth_scan(seed=0), *_synth_imu(t0=0.0))      # bootstrap
    scan = _synth_scan(seed=1)
    t = np.linspace(0.0, 1.0, scan.shape[0])
    r = est.step(scan, *_synth_imu(t0=0.1), scan1_ts=t)     # per-point times accepted
    assert r.pose.shape == (7,) and np.all(np.isfinite(r.pose))
```

- [ ] **Step 2: Run, confirm the estimator test FAILS** (the corrector test should already pass since build_window_sample supports scan1_ts; the estimator test fails because step() has no scan1_ts param)

Run: `cd ros2_ws/src/kiss_imu_ros && KISS_IMU_SRC=/home/mingren/work/3DVision/KISS-IMU/src /home/mingren/anaconda3/envs/air-io/bin/python -m pytest test/test_online_estimator.py::test_step_accepts_scan1_ts -v`
Expected: FAIL (TypeError: step() got an unexpected keyword argument 'scan1_ts').

- [ ] **Step 3: Implement** — in `online_estimator.py`, change the `step` signature and the `build_window_sample` call.

Signature line currently:
```python
    @torch.no_grad()
    def step(self, scan_xyz, accels, gyros, imu_ts) -> OdomResult:
```
becomes:
```python
    @torch.no_grad()
    def step(self, scan_xyz, accels, gyros, imu_ts, scan1_ts=None) -> OdomResult:
```

The `build_window_sample` call currently:
```python
        sample = build_window_sample(accels, gyros, imu_ts, self._prev_scan, scan_xyz)
```
becomes:
```python
        sample = build_window_sample(accels, gyros, imu_ts, self._prev_scan, scan_xyz,
                                     scan1_ts=scan1_ts)
```
Leave everything else (bootstrap branch, device moves, PVGO, anchor advance) unchanged. The bootstrap early-return ignores `scan1_ts`, which is correct.

- [ ] **Step 4: Run full estimator + corrector suites, confirm PASS**

Run: `cd ros2_ws/src/kiss_imu_ros && KISS_IMU_SRC=/home/mingren/work/3DVision/KISS-IMU/src /home/mingren/anaconda3/envs/air-io/bin/python -m pytest test/test_online_estimator.py test/test_imu_corrector.py -v`
Expected: PASS (estimator: 3 existing + 1 new = 4; corrector: 4 existing + 1 new = 5).

- [ ] **Step 5: Commit**

```bash
git add ros2_ws/src/kiss_imu_ros/kiss_imu_ros/online_estimator.py ros2_ws/src/kiss_imu_ros/test/test_online_estimator.py ros2_ws/src/kiss_imu_ros/test/test_imu_corrector.py
git commit -m "feat(ros): plumb per-point scan1_ts through OnlineEstimator.step"
```

---

## Task 4: wire time_field, frame gate, latency, optional downsample into lio_node

**Files:**
- Modify: `ros2_ws/src/kiss_imu_ros/kiss_imu_ros/lio_node.py`
- Modify: `ros2_ws/src/kiss_imu_ros/config/lio.yaml`

This is ROS glue (not unit-testable without rclpy); verify by byte-compile + the existing skipping `test_node_config.py`.

- [ ] **Step 1: Edit `lio_node.py`**

(a) Update imports — change line 13:
```python
from .pointcloud2 import xyz_from_pointcloud2
```
to:
```python
from .pointcloud2 import xyz_from_pointcloud2, voxel_downsample_indexed
```
and add after line 14 (`from .online_estimator import ...`):
```python
from .frame_gate import FrameGate
```
and add `import time` at the top (after `import rclpy` line):
```python
import time
```

(b) In `LioNode.__init__`, extend the `decl` dict — add these keys to it:
```python
            time_field='', max_step_ms=0.0,
```
(put them alongside the existing keys, e.g. right after `voxel_size=0.5,`).

(c) After the existing `self.publish_path = gp('publish_path')` line, add:
```python
        self._time_field = gp('time_field')
        self._voxel_size = float(gp('voxel_size'))
        self._max_step_ms = float(gp('max_step_ms'))
        self._gate = FrameGate()
```

(d) Replace the whole `on_points` method with:
```python
    def on_points(self, msg: PointCloud2):
        if not self._gate.try_enter():
            return                                  # drop backlogged scan, keep latest
        try:
            t = _stamp_to_sec(msg.header.stamp)
            t0 = self._last_scan_t if self._last_scan_t is not None else (t - 0.1)
            ts, acc, gyro = self.buf.pop_window(t0, t)
            self._last_scan_t = t
            scan, scan_t = xyz_from_pointcloud2(msg, time_field=self._time_field or None)
            if self._voxel_size > 0:
                scan, idx = voxel_downsample_indexed(scan, self._voxel_size)
                if scan_t is not None:
                    scan_t = scan_t[idx]            # keep per-point times aligned
            t_start = time.monotonic()
            result = self.estimator.step(scan, acc, gyro, ts, scan1_ts=scan_t)
            dt_ms = (time.monotonic() - t_start) * 1e3
            if self._max_step_ms > 0.0 and dt_ms > self._max_step_ms:
                self.get_logger().warn(
                    f'step took {dt_ms:.0f}ms (> {self._max_step_ms:.0f}ms); '
                    f'dropped so far={self._gate.dropped}')
            self._publish(result, msg.header.stamp)
        finally:
            self._gate.exit()
```

- [ ] **Step 2: Edit `config/lio.yaml`** — add two params and a voxel comment. Under `ros__parameters:` add:
```yaml
    time_field: ""                # per-point time field for deskew; Velodyne: "time", Livox Mid-360: "timestamp". Empty = no deskew.
    max_step_ms: 0.0              # warn if a step() exceeds this (ms); 0 = drop-only, no warning
```
and change the `voxel_size` line to:
```yaml
    voxel_size: 0.0               # node-level voxel downsample; 0 = OFF (recommended: backends voxelize internally; enabling double-downsamples)
```

- [ ] **Step 3: Verify byte-compile + yaml + node test still skips**

Run: `/home/mingren/anaconda3/envs/air-io/bin/python -m py_compile ros2_ws/src/kiss_imu_ros/kiss_imu_ros/lio_node.py && echo "compiles"`
Expected: `compiles`.
Run: `/home/mingren/anaconda3/envs/air-io/bin/python -c "import yaml; yaml.safe_load(open('ros2_ws/src/kiss_imu_ros/config/lio.yaml')); print('yaml OK')"`
Expected: `yaml OK`.
Run: `cd ros2_ws/src/kiss_imu_ros && KISS_IMU_SRC=/home/mingren/work/3DVision/KISS-IMU/src /home/mingren/anaconda3/envs/air-io/bin/python -m pytest test/test_node_config.py -v`
Expected: `1 skipped` (no rclpy) — NOT a collection error.

- [ ] **Step 4: Commit**

```bash
git add ros2_ws/src/kiss_imu_ros/kiss_imu_ros/lio_node.py ros2_ws/src/kiss_imu_ros/config/lio.yaml
git commit -m "feat(ros): wire deskew time_field, frame gate, latency monitor, optional downsample"
```

---

## Task 5: README + full suite + finalize

**Files:**
- Modify: `ros2_ws/src/kiss_imu_ros/README.md`

- [ ] **Step 1: Update README** — make these edits:

(a) In "Must-configure", add a deskew line:
```markdown
- For deskew, set `time_field` to your LiDAR's per-point time field: Velodyne → `time`, Livox Mid-360 → `timestamp` (empty disables deskew). Mid-360 must publish PointCloud2 (not Livox CustomMsg).
```

(b) Add a new section after "Must-configure":
```markdown
## Real-time
- The scan callback uses a non-blocking gate: if a `step()` is still running when a new scan arrives, the new scan is **dropped** (keeps latest, bounds latency). Set `max_step_ms` > 0 to log a warning + cumulative drop count when a step is slow.
- `voxel_size` node-level downsampling defaults to OFF (`0.0`). The ICP backends voxelize internally; enabling node-level downsampling double-downsamples and can hurt accuracy. Only enable it to cap point count for transport/overlap cost; when enabled, per-point deskew times are downsampled with the same indices to stay aligned.
```

(c) In the "Scope (Phase 1)" deferred list: REMOVE the items now implemented in Phase 2a — "per-point LiDAR timestamps for kiss_icp motion compensation", "input voxel downsampling", and the latency/frame-drop item if present. Keep the genuinely-deferred items (sliding-window PVGO, Odometry covariance, high-rate IMU TF, learned correction). Keep edits accurate and minimal.

- [ ] **Step 2: Run the FULL suite (regenerate synth data if needed)**

Run:
```bash
/home/mingren/anaconda3/envs/air-io/bin/python /home/mingren/work/3DVision/KISS-IMU/tools/gen_synth_dataset.py --out /home/mingren/work/3DVision/KISS-IMU/data/synth/Synth01
cd ros2_ws/src/kiss_imu_ros && KISS_IMU_SRC=/home/mingren/work/3DVision/KISS-IMU/src /home/mingren/anaconda3/envs/air-io/bin/python -m pytest test/ -v
```
Expected: all pass + 1 skipped (node test). New counts: pointcloud2 8, frame_gate 2, online_estimator 4, imu_corrector 5, ring_buffer 3, paths 1, replay 1, node_config skipped. Report the summary line. If ANY test FAILS (not skips), STOP and report BLOCKED.

- [ ] **Step 3: Commit**

```bash
git add ros2_ws/src/kiss_imu_ros/README.md
git commit -m "docs(ros): document deskew, frame-drop, voxel interaction; trim deferred list"
```

---

## Self-Review (plan vs spec)

- **Spec coverage:** §2 deskew → Task 1 (normalize + extraction) + Task 3 (thread through step) + Task 4 (node time_field); §3 downsampling → Task 1 (`voxel_downsample_indexed`) + Task 4 (optional, aligned, default off via yaml `0.0`); §4 latency/frame-drop → Task 2 (FrameGate) + Task 4 (wiring + `max_step_ms`); §5 reuse/new + backward-compat → `step(scan1_ts=None)` default keeps Phase 1 callers working; §6 testing → pure-function tests in Tasks 1-3, byte-compile in Task 4, full suite in Task 5; §7 risks → README notes (kiss_icp semantics single change-point = `normalize_per_point_times`; Mid-360 PointCloud2 caveat documented).
- **Placeholder scan:** none — every code step has full content.
- **Type consistency:** `xyz_from_pointcloud2` returns `(xyz, t_norm_or_None)` (Task 1) and the node unpacks `scan, scan_t` (Task 1 call-site fix + Task 4); `voxel_downsample_indexed` returns `(pts, idx)` used in Task 4; `step(..., scan1_ts=None)` (Task 3) called with `scan1_ts=scan_t` (Task 4); `FrameGate.try_enter/exit/dropped` (Task 2) used in Task 4. Consistent.
- **Backward compatibility:** all signature additions are keyword args with defaults; existing Phase 1 tests untouched and must stay green (verified in Task 5 full run).
