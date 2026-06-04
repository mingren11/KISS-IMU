# 实时 ROS2 LIO 前端（Phase 1）实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 KISS-IMU 的离线 LIO 管线做成一个能在 ROS2 + GPU 真机上实时运行的里程计节点；Phase 1 用 raw IMU（无学习模型）+ ICP + 2 节点 PVGO 跑通端到端 odom 输出。

**Architecture:** 两层——纯 Python、无 ROS 依赖的 `OnlineEstimator`（由 `src/inference.py` 的逐窗口循环重构而来，每步处理一帧点云 + 区间 IMU），外加一个薄 rclpy 节点 `lio_node` 负责订阅 `/imu`、`/points`，缓冲 IMU、切窗口、发布 `nav_msgs/Odometry` + TF。IMU 修正级做成可插拔策略，Phase 1 用恒等的 `RawCorrector`。

**Tech Stack:** Python 3.10、ROS2（rclpy）、PyTorch、pypose、small_gicp/kiss_icp（LO 后端）、pytest。

设计依据：`docs/superpowers/specs/2026-06-04-realtime-ros2-lio-frontend-design.md`。

---

## 约定与前置

- 仓库根：`/home/mingren/work/3DVision/KISS-IMU`（下文 `<repo>`）。
- 上游算法包在 `<repo>/src`，模块以 `training.*` / `models.*` / `data.*` 顶层导入（和 `scripts/train.sh` 一致：从 `src/` 下运行）。新代码通过把 `<repo>/src` 加入 `sys.path` 来复用，**不修改 `src/` 主体**。
- 新 ROS2 包：`<repo>/ros2_ws/src/kiss_imu_ros/`。
- 测试统一用 `pytest`，可在纯 CPU、无 ROS、无传感器下跑（LO 后端用 pip 可装的 `small_gicp`）。
- 每个 Task 结束都提交。提交信息结尾加
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`。

---

## File Structure

| 文件 | 职责 |
| --- | --- |
| `ros2_ws/src/kiss_imu_ros/package.xml`、`setup.py`、`setup.cfg`、`resource/kiss_imu_ros` | ament_python 包元数据 |
| `ros2_ws/src/kiss_imu_ros/kiss_imu_ros/__init__.py` | 包初始化 |
| `ros2_ws/src/kiss_imu_ros/kiss_imu_ros/paths.py` | 把 `<repo>/src` 加入 `sys.path`（KISS_IMU_SRC 环境变量可覆盖） |
| `ros2_ws/src/kiss_imu_ros/kiss_imu_ros/ring_buffer.py` | 线程安全 IMU 环形缓冲，按时间区间取窗口 |
| `ros2_ws/src/kiss_imu_ros/kiss_imu_ros/pointcloud2.py` | `PointCloud2` ↔ `(N,3)` numpy |
| `ros2_ws/src/kiss_imu_ros/kiss_imu_ros/imu_corrector.py` | `RawCorrector`（恒等）+ `build_window_sample()` |
| `ros2_ws/src/kiss_imu_ros/kiss_imu_ros/online_estimator.py` | `OnlineEstimator.step()`：核心融合 |
| `ros2_ws/src/kiss_imu_ros/kiss_imu_ros/lio_node.py` | rclpy 节点：I/O、TF、executor |
| `ros2_ws/src/kiss_imu_ros/config/lio.yaml` | 参数：外参、重力、topic、lm_weight 等 |
| `ros2_ws/src/kiss_imu_ros/launch/lio.launch.py` | 启动节点 + 载入参数 |
| `ros2_ws/src/kiss_imu_ros/test/test_*.py` | 单测 |
| `ros2_ws/src/kiss_imu_ros/README.md` | 构建/运行/回放说明 |

---

## Task 0: 脚手架 ROS2 包

**Files:**
- Create: `ros2_ws/src/kiss_imu_ros/package.xml`
- Create: `ros2_ws/src/kiss_imu_ros/setup.py`
- Create: `ros2_ws/src/kiss_imu_ros/setup.cfg`
- Create: `ros2_ws/src/kiss_imu_ros/resource/kiss_imu_ros`
- Create: `ros2_ws/src/kiss_imu_ros/kiss_imu_ros/__init__.py`
- Create: `ros2_ws/src/kiss_imu_ros/kiss_imu_ros/paths.py`
- Test: `ros2_ws/src/kiss_imu_ros/test/test_paths.py`

- [ ] **Step 1: 写包元数据**

`package.xml`：
```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>kiss_imu_ros</name>
  <version>0.1.0</version>
  <description>Real-time ROS2 LiDAR-inertial odometry front-end built on KISS-IMU.</description>
  <maintainer email="shuaiqiduoyi@gmail.com">Ming Ren</maintainer>
  <license>MIT</license>

  <exec_depend>rclpy</exec_depend>
  <exec_depend>sensor_msgs</exec_depend>
  <exec_depend>nav_msgs</exec_depend>
  <exec_depend>geometry_msgs</exec_depend>
  <exec_depend>tf2_ros</exec_depend>
  <exec_depend>std_msgs</exec_depend>

  <test_depend>ament_copyright</test_depend>
  <test_depend>ament_flake8</test_depend>
  <test_depend>python3-pytest</test_depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
```

`setup.py`：
```python
import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'kiss_imu_ros'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Ming Ren',
    maintainer_email='shuaiqiduoyi@gmail.com',
    description='Real-time ROS2 LiDAR-inertial odometry front-end built on KISS-IMU.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'lio_node = kiss_imu_ros.lio_node:main',
        ],
    },
)
```

`setup.cfg`：
```ini
[develop]
script_dir=$base/lib/kiss_imu_ros
[install]
install_scripts=$base/lib/kiss_imu_ros
```

`resource/kiss_imu_ros`：空文件。
`kiss_imu_ros/__init__.py`：空文件。

- [ ] **Step 2: 写 paths.py（先写失败测试）**

`test/test_paths.py`：
```python
import importlib


def test_ensure_src_on_path_imports_upstream(monkeypatch, tmp_path):
    # KISS_IMU_SRC 指向真实仓库 src，确保上游模块可导入
    import kiss_imu_ros.paths as paths
    paths.ensure_src_on_path()
    # 上游核心模块应能导入
    assert importlib.import_module('models.imu_net') is not None
    assert importlib.import_module('training.integrator') is not None
```

- [ ] **Step 3: 跑测试确认失败**

Run: `cd ros2_ws/src/kiss_imu_ros && KISS_IMU_SRC=<repo>/src python -m pytest test/test_paths.py -v`
Expected: FAIL（`No module named 'kiss_imu_ros.paths'`）

- [ ] **Step 4: 实现 paths.py**

```python
"""Make the upstream KISS-IMU `src/` importable (training.*, models.*, data.*).

The algorithm code lives in <repo>/src and is imported with top-level package
names, exactly as scripts/train.sh runs it (`cd src; python train.py`). We add
that directory to sys.path instead of copying or modifying it.
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
```

- [ ] **Step 5: 跑测试确认通过**

Run: `cd ros2_ws/src/kiss_imu_ros && KISS_IMU_SRC=<repo>/src python -m pytest test/test_paths.py -v`
Expected: PASS

- [ ] **Step 6: 提交**

```bash
git add ros2_ws/src/kiss_imu_ros
git commit -m "feat(ros): scaffold kiss_imu_ros package + src path bootstrap"
```

---

## Task 1: 线程安全 IMU 环形缓冲

**Files:**
- Create: `ros2_ws/src/kiss_imu_ros/kiss_imu_ros/ring_buffer.py`
- Test: `ros2_ws/src/kiss_imu_ros/test/test_ring_buffer.py`

`ImuBuffer` 存 `(t, ax,ay,az, gx,gy,gz)`，提供 `append()` 和 `pop_window(t0, t1)`：返回时间在 `(t0, t1]` 的样本（按时间升序），并丢弃 `<= t1` 的旧样本。线程安全。

- [ ] **Step 1: 写失败测试**

`test/test_ring_buffer.py`：
```python
import numpy as np
from kiss_imu_ros.ring_buffer import ImuBuffer


def test_pop_window_returns_interval_inclusive_upper():
    buf = ImuBuffer(maxlen=100)
    for i in range(10):
        buf.append(t=float(i), acc=(i, 0.0, 0.0), gyro=(0.0, 0.0, float(i)))
    ts, acc, gyro = buf.pop_window(t0=2.0, t1=5.0)
    # (2,5] -> t = 3,4,5
    assert ts.tolist() == [3.0, 4.0, 5.0]
    assert acc.shape == (3, 3) and gyro.shape == (3, 3)
    assert np.allclose(acc[:, 0], [3.0, 4.0, 5.0])


def test_pop_window_discards_consumed_samples():
    buf = ImuBuffer(maxlen=100)
    for i in range(6):
        buf.append(t=float(i), acc=(0.0, 0.0, 0.0), gyro=(0.0, 0.0, 0.0))
    buf.pop_window(t0=-1.0, t1=2.0)   # 消费 t=0,1,2
    ts, _, _ = buf.pop_window(t0=2.0, t1=10.0)
    assert ts.tolist() == [3.0, 4.0, 5.0]


def test_pop_window_empty_when_no_samples_in_range():
    buf = ImuBuffer(maxlen=100)
    buf.append(t=0.0, acc=(0.0, 0.0, 0.0), gyro=(0.0, 0.0, 0.0))
    ts, acc, gyro = buf.pop_window(t0=10.0, t1=20.0)
    assert ts.shape == (0,) and acc.shape == (0, 3) and gyro.shape == (0, 3)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd ros2_ws/src/kiss_imu_ros && python -m pytest test/test_ring_buffer.py -v`
Expected: FAIL（`No module named 'kiss_imu_ros.ring_buffer'`）

- [ ] **Step 3: 实现 ring_buffer.py**

```python
"""Thread-safe IMU ring buffer with time-window extraction."""
import threading
from collections import deque

import numpy as np


class ImuBuffer:
    def __init__(self, maxlen: int = 4000):
        self._buf = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def append(self, t: float, acc, gyro) -> None:
        with self._lock:
            self._buf.append((float(t),
                              float(acc[0]), float(acc[1]), float(acc[2]),
                              float(gyro[0]), float(gyro[1]), float(gyro[2])))

    def pop_window(self, t0: float, t1: float):
        """Return samples with t in (t0, t1] sorted ascending; drop samples <= t1."""
        with self._lock:
            rows = [r for r in self._buf if t0 < r[0] <= t1]
            # drop everything up to and including t1
            kept = deque((r for r in self._buf if r[0] > t1), maxlen=self._buf.maxlen)
            self._buf = kept
        if not rows:
            return (np.zeros((0,), dtype=np.float64),
                    np.zeros((0, 3), dtype=np.float64),
                    np.zeros((0, 3), dtype=np.float64))
        rows.sort(key=lambda r: r[0])
        arr = np.asarray(rows, dtype=np.float64)
        return arr[:, 0].copy(), arr[:, 1:4].copy(), arr[:, 4:7].copy()
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd ros2_ws/src/kiss_imu_ros && python -m pytest test/test_ring_buffer.py -v`
Expected: PASS（3 passed）

- [ ] **Step 5: 提交**

```bash
git add ros2_ws/src/kiss_imu_ros/kiss_imu_ros/ring_buffer.py ros2_ws/src/kiss_imu_ros/test/test_ring_buffer.py
git commit -m "feat(ros): thread-safe IMU ring buffer with time-window pop"
```

---

## Task 2: PointCloud2 ↔ numpy 转换

**Files:**
- Create: `ros2_ws/src/kiss_imu_ros/kiss_imu_ros/pointcloud2.py`
- Test: `ros2_ws/src/kiss_imu_ros/test/test_pointcloud2.py`

提供 `xyz_from_arrays(structured)`（把结构化数组取 x/y/z 成 `(N,3) float64`，过滤 NaN/Inf）和 `voxel_downsample(pts, voxel)`（确定性体素降采样，复用上游 `_voxel_downsample_np` 算法）。`PointCloud2` 的解析在节点里用 `sensor_msgs_py.point_cloud2.read_points`，这里只测纯 numpy 逻辑（无 ROS 依赖）。

- [ ] **Step 1: 写失败测试**

`test/test_pointcloud2.py`：
```python
import numpy as np
from kiss_imu_ros.pointcloud2 import xyz_from_arrays, voxel_downsample


def test_xyz_from_structured_filters_nonfinite():
    dt = np.dtype([('x', np.float32), ('y', np.float32), ('z', np.float32), ('intensity', np.float32)])
    a = np.zeros((4,), dtype=dt)
    a['x'] = [1.0, np.nan, 3.0, np.inf]
    a['y'] = [1.0, 2.0, 3.0, 4.0]
    a['z'] = [1.0, 2.0, 3.0, 4.0]
    pts = xyz_from_arrays(a)
    assert pts.shape == (2, 3) and pts.dtype == np.float64
    assert np.allclose(pts[:, 0], [1.0, 3.0])


def test_voxel_downsample_reduces_duplicates():
    pts = np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [5.0, 5.0, 5.0]], dtype=np.float64)
    out = voxel_downsample(pts, voxel=1.0)
    assert out.shape[0] == 2   # 前两个落同一体素
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd ros2_ws/src/kiss_imu_ros && python -m pytest test/test_pointcloud2.py -v`
Expected: FAIL（`No module named 'kiss_imu_ros.pointcloud2'`）

- [ ] **Step 3: 实现 pointcloud2.py**

```python
"""PointCloud2 <-> numpy helpers (pure-numpy parts are unit-tested without ROS)."""
import numpy as np


def xyz_from_arrays(structured) -> np.ndarray:
    """Take x/y/z from a structured array; return (N,3) float64 with NaN/Inf dropped."""
    xyz = np.stack([structured['x'], structured['y'], structured['z']], axis=-1).astype(np.float64)
    mask = np.isfinite(xyz).all(axis=1)
    return xyz[mask]


def voxel_downsample(pts: np.ndarray, voxel: float) -> np.ndarray:
    """Deterministic voxel downsample (unique grid cell, sorted) — mirrors upstream."""
    if pts.size == 0 or voxel <= 0:
        return pts
    grid = np.floor(pts / voxel).astype(np.int64)
    _, idx = np.unique(grid, axis=0, return_index=True)
    return pts[np.sort(idx)]


def xyz_from_pointcloud2(msg, field_names=('x', 'y', 'z')) -> np.ndarray:
    """Parse a sensor_msgs/PointCloud2 into (N,3) float64. Imports ROS lazily."""
    from sensor_msgs_py import point_cloud2
    structured = point_cloud2.read_points(msg, field_names=field_names, skip_nans=True)
    xyz = np.stack([structured['x'], structured['y'], structured['z']], axis=-1).astype(np.float64)
    return xyz
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd ros2_ws/src/kiss_imu_ros && python -m pytest test/test_pointcloud2.py -v`
Expected: PASS（2 passed）

- [ ] **Step 5: 提交**

```bash
git add ros2_ws/src/kiss_imu_ros/kiss_imu_ros/pointcloud2.py ros2_ws/src/kiss_imu_ros/test/test_pointcloud2.py
git commit -m "feat(ros): PointCloud2<->numpy + deterministic voxel downsample"
```

---

## Task 3: RawCorrector + 窗口样本构造

**Files:**
- Create: `ros2_ws/src/kiss_imu_ros/kiss_imu_ros/imu_corrector.py`
- Test: `ros2_ws/src/kiss_imu_ros/test/test_imu_corrector.py`

`build_window_sample(...)` 直接构造 B=1 的样本字典（不经 `collate_fn`，避免它要求 GT 键），只包含下游 `IMUNet.forward` 和 `LOModule.forward` 实际读取的键。`RawCorrector` 输出与 `IMUNet.forward` **同结构**的 `corr_data`（见 `src/models/imu_net.py:136-143`）：`accels_corr`/`gyros_corr`/`dts` 为长度 1 的 list，`acc_cov`/`gyr_cov` 为 None，`valid_length` 为 `(1,)` long tensor。

- [ ] **Step 1: 写失败测试**

`test/test_imu_corrector.py`：
```python
import numpy as np
import torch
from kiss_imu_ros.imu_corrector import build_window_sample, RawCorrector


def _fake_window():
    T = 20
    accels = np.random.randn(T, 3).astype(np.float64)
    gyros = np.random.randn(T, 3).astype(np.float64)
    imu_ts = np.linspace(0.0, 0.1, T)
    scan0 = np.random.randn(500, 3).astype(np.float64)
    scan1 = np.random.randn(500, 3).astype(np.float64)
    return accels, gyros, imu_ts, scan0, scan1


def test_build_window_sample_shapes_and_keys():
    accels, gyros, imu_ts, scan0, scan1 = _fake_window()
    s = build_window_sample(accels, gyros, imu_ts, scan0, scan1)
    assert s['accels'].shape == (1, 20, 3)
    assert s['gyros'].shape == (1, 20, 3)
    assert s['imu_dts'].shape == (1, 20)
    assert s['valid_length'].tolist() == [20]
    assert isinstance(s['scan0'], list) and s['scan0'][0].shape == (500, 3)
    assert len(s['scan1']) == 1 and len(s['scan1_ts']) == 1


def test_raw_corrector_passthrough_structure():
    accels, gyros, imu_ts, scan0, scan1 = _fake_window()
    s = build_window_sample(accels, gyros, imu_ts, scan0, scan1)
    corr = RawCorrector().correct(s)
    assert set(['accels_corr', 'gyros_corr', 'acc_cov', 'gyr_cov', 'valid_length', 'dts']) <= set(corr)
    assert len(corr['accels_corr']) == 1 and corr['accels_corr'][0].shape == (20, 3)
    assert corr['acc_cov'] is None and corr['gyr_cov'] is None
    # raw = 不修正：透传等于输入
    assert torch.allclose(corr['accels_corr'][0].cpu().double(),
                          torch.from_numpy(accels))
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd ros2_ws/src/kiss_imu_ros && python -m pytest test/test_imu_corrector.py -v`
Expected: FAIL（`No module named 'kiss_imu_ros.imu_corrector'`）

- [ ] **Step 3: 实现 imu_corrector.py**

```python
"""Pluggable IMU correction stage.

RawCorrector = identity passthrough (the raw_pvgo baseline). A future
LearnedCorrector will wrap IMUNet and return the same dict structure
(see src/models/imu_net.py forward()), so OnlineEstimator stays unchanged.
"""
import numpy as np
import torch


def build_window_sample(accels, gyros, imu_ts, scan0, scan1, scan1_ts=None, device='cpu'):
    """Build a batch-of-1 sample dict with exactly the keys IMUNet.forward and
    LOModule.forward read. dts[k] = ts[k]-ts[k-1], with dts[0] duplicated from dts[1]."""
    accels = np.asarray(accels, dtype=np.float64)
    gyros = np.asarray(gyros, dtype=np.float64)
    imu_ts = np.asarray(imu_ts, dtype=np.float64)
    T = accels.shape[0]
    if T >= 2:
        dts = np.empty(T, dtype=np.float64)
        dts[1:] = np.diff(imu_ts)
        dts[0] = dts[1]
    else:
        dts = np.full(T, 1e-3, dtype=np.float64)

    a = torch.from_numpy(accels).float().unsqueeze(0)        # (1,T,3)
    g = torch.from_numpy(gyros).float().unsqueeze(0)         # (1,T,3)
    ts = torch.from_numpy(imu_ts).float().unsqueeze(0)       # (1,T)
    dt = torch.from_numpy(dts).float().unsqueeze(0)          # (1,T)
    s0 = torch.from_numpy(np.asarray(scan0, dtype=np.float64)).float()
    s1 = torch.from_numpy(np.asarray(scan1, dtype=np.float64)).float()
    if scan1_ts is None:
        scan1_ts = np.zeros(s1.shape[0], dtype=np.float64)
    s1ts = torch.from_numpy(np.asarray(scan1_ts, dtype=np.float64)).float()

    return {
        'accels': a, 'gyros': g, 'imu_ts': ts, 'imu_dts': dt,
        'valid_length': torch.tensor([T], dtype=torch.long),
        'scan0': [s0], 'scan1': [s1], 'scan1_ts': [s1ts],
    }


class RawCorrector:
    """Identity: emit corrected IMU == raw IMU, no covariance."""

    def correct(self, sample) -> dict:
        vlen = int(sample['valid_length'][0].item())
        return {
            'accels_corr': [sample['accels'][0, :vlen]],
            'gyros_corr': [sample['gyros'][0, :vlen]],
            'acc_cov': None,
            'gyr_cov': None,
            'valid_length': sample['valid_length'],
            'dts': [sample['imu_dts'][0, :vlen]],
        }
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd ros2_ws/src/kiss_imu_ros && python -m pytest test/test_imu_corrector.py -v`
Expected: PASS（2 passed）

- [ ] **Step 5: 提交**

```bash
git add ros2_ws/src/kiss_imu_ros/kiss_imu_ros/imu_corrector.py ros2_ws/src/kiss_imu_ros/test/test_imu_corrector.py
git commit -m "feat(ros): RawCorrector + batch-of-1 window sample builder"
```

---

## Task 4: OnlineEstimator.step（核心）

**Files:**
- Create: `ros2_ws/src/kiss_imu_ros/kiss_imu_ros/online_estimator.py`
- Test: `ros2_ws/src/kiss_imu_ros/test/test_online_estimator.py`

由 `src/inference.py:99-159` 循环体重构。每个 `step(scan_xyz, accels, gyros, imu_ts)`：
1. `RawCorrector.correct` → corrected IMU；
2. `IMUIntegrator.integrate`（`motion_mode=False` 得全局节点，`motion_mode=True` 得相对运动+协方差）；
3. `LOModule.forward(sample, anchor)` → ICP 相对运动 + overlap；
4. `pvgo.optimize`（2 节点，常量 `lm_weight`）→ 优化位姿/速度；
5. 推进 `anchor_pose`/`anchor_vel`，返回 `OdomResult`。

第一帧 bootstrap：存 scan、返回初始位姿、不做配准（无 scan0）。

后端默认 `small_gicp`（pip 可装、CPU 可跑）；真机用 `kiss_icp`。

- [ ] **Step 1: 写失败测试（合成数据冒烟）**

`test/test_online_estimator.py`：
```python
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pypose")
pytest.importorskip("small_gicp")

from kiss_imu_ros.online_estimator import OnlineEstimator, EstimatorConfig


def _synth_scan(n=800, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(-20, 20, size=(n, 3)).astype(np.float64)


def _synth_imu(T=20, t0=0.0, dt=0.005):
    ts = t0 + np.arange(T) * dt
    accels = np.tile([0.0, 0.0, 9.81], (T, 1)).astype(np.float64)  # 静止：仅抵消重力
    gyros = np.zeros((T, 3), dtype=np.float64)
    return accels, gyros, ts


def test_step_returns_finite_se3_pose():
    cfg = EstimatorConfig(lo_model='small_gicp', device='cpu')
    est = OnlineEstimator(cfg)
    # 第一帧 bootstrap
    r0 = est.step(_synth_scan(seed=0), *_synth_imu(t0=0.0))
    assert r0.pose.shape == (7,)
    # 第二帧：应产出有限位姿
    r1 = est.step(_synth_scan(seed=1), *_synth_imu(t0=0.1))
    assert r1.pose.shape == (7,)
    assert np.all(np.isfinite(r1.pose))
    # 四元数近似单位范数
    q = r1.pose[3:]
    assert abs(np.linalg.norm(q) - 1.0) < 1e-3


def test_pose_advances_over_steps():
    cfg = EstimatorConfig(lo_model='small_gicp', device='cpu')
    est = OnlineEstimator(cfg)
    poses = []
    for k in range(4):
        r = est.step(_synth_scan(seed=k), *_synth_imu(t0=0.1 * k))
        poses.append(r.pose[:3])
    # 位姿序列应可计算且有限（不强求精度，只验证管线连贯）
    assert all(np.all(np.isfinite(p)) for p in poses)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd ros2_ws/src/kiss_imu_ros && KISS_IMU_SRC=<repo>/src python -m pytest test/test_online_estimator.py -v`
Expected: FAIL（`No module named 'kiss_imu_ros.online_estimator'`）

- [ ] **Step 3: 实现 online_estimator.py**

```python
"""Streaming LiDAR-inertial odometry estimator (ROS-free).

Refactored from src/inference.py's per-window loop. One step() == one scan +
the IMU interval that precedes it. Phase 1 fuses with a 2-node PVGO and a raw
(identity) IMU corrector; the corrector is pluggable for a learned model later.
"""
from dataclasses import dataclass, field

import numpy as np
import torch

from .paths import ensure_src_on_path
ensure_src_on_path()

import pypose as pp                                  # noqa: E402
from training.integrator import IMUIntegrator        # noqa: E402
from models.lo_module import LOModule                # noqa: E402
from models.pvgo import optimize                     # noqa: E402

from .imu_corrector import build_window_sample, RawCorrector  # noqa: E402


@dataclass
class EstimatorConfig:
    lo_model: str = 'kiss_icp'
    device: str = 'cuda:0'
    use_submap: bool = False
    lm_weight: tuple = (1.0, 0.1, 1.0, 0.1, 0.1)
    gravity: tuple = (0.0, 0.0, 9.81)
    # IMU->LiDAR extrinsic (defaults = identity; MUST be set per real robot)
    R_I_L: list = field(default_factory=lambda: [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    T_I_L: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    init_pose: tuple = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)  # xyz + quat xyzw
    init_vel: tuple = (0.0, 0.0, 0.0)


@dataclass
class OdomResult:
    pose: np.ndarray       # (7,) xyz + quat xyzw
    vel: np.ndarray        # (3,)
    overlap: float


class OnlineEstimator:
    def __init__(self, cfg: EstimatorConfig):
        self.cfg = cfg
        self.device = cfg.device
        self.gravity = torch.tensor(list(cfg.gravity), dtype=torch.float32)
        self.lm_weight = list(cfg.lm_weight)
        self.corrector = RawCorrector()

        init_pose7 = torch.tensor(list(cfg.init_pose), dtype=torch.float32)
        init_state = {
            'pos': torch.tensor(list(cfg.init_pose[:3]), dtype=torch.float32),
            'rot': torch.tensor(list(cfg.init_pose[3:]), dtype=torch.float32),
            'vel': torch.tensor(list(cfg.init_vel), dtype=torch.float32),
        }
        self.integrator = IMUIntegrator(init_state=init_state,
                                        gravity=self.gravity, device=self.device)
        self.lo_model = LOModule(
            lo_model=cfg.lo_model,
            T_I_L=np.asarray(cfg.T_I_L, dtype=np.float64),
            R_I_L=np.asarray(cfg.R_I_L, dtype=np.float64),
            init_state=init_pose7,
            device_id=self.device,
            use_submap=cfg.use_submap,
        )

        self.anchor_pose = init_pose7.clone()
        self.anchor_vel = torch.tensor(list(cfg.init_vel), dtype=torch.float32)
        self._prev_scan = None

    @torch.no_grad()
    def step(self, scan_xyz, accels, gyros, imu_ts) -> OdomResult:
        scan_xyz = np.asarray(scan_xyz, dtype=np.float64)

        # bootstrap: no previous scan -> just store and report current anchor
        if self._prev_scan is None or len(accels) < 2:
            self._prev_scan = scan_xyz
            return OdomResult(pose=self.anchor_pose.numpy().copy(),
                              vel=self.anchor_vel.numpy().copy(), overlap=0.0)

        sample = build_window_sample(accels, gyros, imu_ts,
                                      self._prev_scan, scan_xyz)
        corr = self.corrector.correct(sample)

        init_pos = self.anchor_pose[:3].to(self.device).float()
        init_rot = self.anchor_pose[3:].to(self.device).float()
        init_rot = init_rot / (torch.linalg.norm(init_rot) + 1e-12)
        init_v = self.anchor_vel.to(self.device).float()
        init_state = {'rot': pp.SO3(init_rot), 'vel': init_v, 'pos': init_pos, 'cov': None}

        imu_states = self.integrator.integrate(
            init=init_state, dts=corr['dts'],
            accels=corr['accels_corr'], gyros=corr['gyros_corr'],
            cov_accels=corr['acc_cov'], cov_gyros=corr['gyr_cov'],
            motion_mode=False)
        zero_init = {'rot': pp.SO3(init_rot),
                     'vel': torch.zeros((1, 3), device=self.device),
                     'pos': torch.zeros((1, 3), device=self.device), 'cov': None}
        imu_d = self.integrator.integrate(
            init=zero_init, dts=corr['dts'],
            accels=corr['accels_corr'], gyros=corr['gyros_corr'],
            cov_accels=corr['acc_cov'], cov_gyros=corr['gyr_cov'],
            motion_mode=True)

        imu_nodes = pp.SE3(torch.cat([imu_states['pos'], imu_states['rot'].tensor()], dim=-1)).to(self.device)
        imu_vels = imu_states['vel'].to(self.device)
        imu_dts = torch.stack([d.sum() for d in corr['dts']]).unsqueeze(-1).to(self.device)

        icp_poses, icp_motions, icp_overlap = self.lo_model(
            sample, pp.SE3(self.anchor_pose.to(self.device)))

        pgo_poses, pgo_vels = optimize(
            nodes=imu_nodes, vels=imu_vels,
            icp_factors=icp_motions,
            imu_drots=imu_d['rot'], imu_dvels=imu_d['vel'],
            imu_dtrans=imu_d['pos'], imu_dts=imu_dts,
            weights=self.lm_weight, gravity=self.gravity,
            icp_weights=None, imu_weights=None, device=self.device)

        self.anchor_pose = pgo_poses.tensor()[-1].detach().cpu()
        self.anchor_vel = pgo_vels[-1].detach().cpu()
        self._prev_scan = scan_xyz

        return OdomResult(
            pose=self.anchor_pose.numpy().copy(),
            vel=self.anchor_vel.numpy().copy(),
            overlap=float(np.asarray(icp_overlap).reshape(-1)[-1]),
        )
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd ros2_ws/src/kiss_imu_ros && KISS_IMU_SRC=<repo>/src python -m pytest test/test_online_estimator.py -v`
Expected: PASS（2 passed）。若 `small_gicp` 未装则 skip。

- [ ] **Step 5: 提交**

```bash
git add ros2_ws/src/kiss_imu_ros/kiss_imu_ros/online_estimator.py ros2_ws/src/kiss_imu_ros/test/test_online_estimator.py
git commit -m "feat(ros): OnlineEstimator.step — streaming IMU+ICP+2-node PVGO"
```

---

## Task 5: 配置文件与 launch

**Files:**
- Create: `ros2_ws/src/kiss_imu_ros/config/lio.yaml`
- Create: `ros2_ws/src/kiss_imu_ros/launch/lio.launch.py`

- [ ] **Step 1: 写 config/lio.yaml**

```yaml
/lio_node:
  ros__parameters:
    imu_topic: "/imu"
    points_topic: "/points"
    odom_topic: "odometry"
    odom_frame: "odom"
    base_frame: "base_link"
    lo_model: "kiss_icp"          # 真机推荐 kiss_icp
    use_submap: false
    device: "cuda:0"
    lm_weight: [1.0, 0.1, 1.0, 0.1, 0.1]
    gravity: [0.0, 0.0, 9.81]
    # !! 必须按实机标定填写 IMU->LiDAR 外参，不能用默认单位阵 !!
    R_I_L: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]   # 行主序 3x3
    T_I_L: [0.0, 0.0, 0.0]
    voxel_size: 0.5
    publish_path: true
```

- [ ] **Step 2: 写 launch/lio.launch.py**

```python
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    cfg = os.path.join(get_package_share_directory('kiss_imu_ros'), 'config', 'lio.yaml')
    return LaunchDescription([
        Node(
            package='kiss_imu_ros',
            executable='lio_node',
            name='lio_node',
            output='screen',
            parameters=[cfg],
        ),
    ])
```

- [ ] **Step 3: 提交**

```bash
git add ros2_ws/src/kiss_imu_ros/config ros2_ws/src/kiss_imu_ros/launch
git commit -m "feat(ros): lio config + launch"
```

---

## Task 6: lio_node（rclpy 节点）

**Files:**
- Create: `ros2_ws/src/kiss_imu_ros/kiss_imu_ros/lio_node.py`

订阅 `/imu`（入缓冲）、`/points`（切窗口→`estimator.step()`→发布 odom + TF）。用 `MultiThreadedExecutor` + 两个互斥回调组，IMU 回调只入缓冲，点云回调跑重活。ROS 节点无法纯单测，靠 Task 7 的 bag 回放验证；本 Task 只保证 import 干净、参数读取正确。

- [ ] **Step 1: 写节点冒烟测试（不启动 rclpy，只校验模块可导入 + 参数→cfg 映射函数）**

`test/test_node_config.py`：
```python
import pytest
pytest.importorskip("torch")
pytest.importorskip("pypose")


def test_params_to_cfg_maps_extrinsics():
    from kiss_imu_ros.lio_node import params_to_cfg
    params = dict(lo_model='small_gicp', device='cpu', use_submap=False,
                  lm_weight=[1.0, 0.1, 1.0, 0.1, 0.1], gravity=[0.0, 0.0, 9.81],
                  R_I_L=[1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0], T_I_L=[0.1, 0.2, 0.3])
    cfg = params_to_cfg(params)
    assert cfg.lo_model == 'small_gicp'
    assert cfg.R_I_L == [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]
    assert cfg.T_I_L == [0.1, 0.2, 0.3]
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd ros2_ws/src/kiss_imu_ros && KISS_IMU_SRC=<repo>/src python -m pytest test/test_node_config.py -v`
Expected: FAIL（`No module named 'kiss_imu_ros.lio_node'` 或 `params_to_cfg` 未定义）

- [ ] **Step 3: 实现 lio_node.py**

```python
"""ROS2 node: subscribe IMU + PointCloud2, publish Odometry + TF."""
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu, PointCloud2
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped, TransformStamped
from tf2_ros import TransformBroadcaster

from .ring_buffer import ImuBuffer
from .pointcloud2 import xyz_from_pointcloud2
from .online_estimator import OnlineEstimator, EstimatorConfig


def _stamp_to_sec(stamp) -> float:
    return stamp.sec + stamp.nanosec * 1e-9


def params_to_cfg(p: dict) -> EstimatorConfig:
    R = list(p['R_I_L'])
    R3 = [R[0:3], R[3:6], R[6:9]]
    return EstimatorConfig(
        lo_model=p['lo_model'], device=p['device'], use_submap=bool(p['use_submap']),
        lm_weight=tuple(p['lm_weight']), gravity=tuple(p['gravity']),
        R_I_L=R3, T_I_L=list(p['T_I_L']),
    )


class LioNode(Node):
    def __init__(self):
        super().__init__('lio_node')
        decl = dict(
            imu_topic='/imu', points_topic='/points', odom_topic='odometry',
            odom_frame='odom', base_frame='base_link', lo_model='kiss_icp',
            use_submap=False, device='cuda:0', lm_weight=[1.0, 0.1, 1.0, 0.1, 0.1],
            gravity=[0.0, 0.0, 9.81], R_I_L=[1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0],
            T_I_L=[0.0, 0.0, 0.0], voxel_size=0.5, publish_path=True)
        for k, v in decl.items():
            self.declare_parameter(k, v)
        gp = lambda k: self.get_parameter(k).value
        self.odom_frame = gp('odom_frame'); self.base_frame = gp('base_frame')
        self.publish_path = gp('publish_path')

        self.buf = ImuBuffer()
        self.estimator = OnlineEstimator(params_to_cfg({k: gp(k) for k in
            ['lo_model', 'device', 'use_submap', 'lm_weight', 'gravity', 'R_I_L', 'T_I_L']}))
        self._last_scan_t = None

        imu_cb = MutuallyExclusiveCallbackGroup()
        pts_cb = MutuallyExclusiveCallbackGroup()
        self.create_subscription(Imu, gp('imu_topic'), self.on_imu,
                                 qos_profile_sensor_data, callback_group=imu_cb)
        self.create_subscription(PointCloud2, gp('points_topic'), self.on_points,
                                 qos_profile_sensor_data, callback_group=pts_cb)
        self.odom_pub = self.create_publisher(Odometry, gp('odom_topic'), 10)
        self.path_pub = self.create_publisher(Path, 'path', 10) if self.publish_path else None
        self.tf_bc = TransformBroadcaster(self)
        self._path = Path(); self._path.header.frame_id = self.odom_frame
        self.get_logger().info('lio_node ready')

    def on_imu(self, msg: Imu):
        t = _stamp_to_sec(msg.header.stamp)
        self.buf.append(t,
                        (msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z),
                        (msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z))

    def on_points(self, msg: PointCloud2):
        t = _stamp_to_sec(msg.header.stamp)
        t0 = self._last_scan_t if self._last_scan_t is not None else (t - 0.1)
        ts, acc, gyro = self.buf.pop_window(t0, t)
        self._last_scan_t = t
        scan = xyz_from_pointcloud2(msg)
        result = self.estimator.step(scan, acc, gyro, ts)
        self._publish(result, msg.header.stamp)

    def _publish(self, result, stamp):
        p, q = result.pose[:3], result.pose[3:]
        odom = Odometry()
        odom.header.stamp = stamp; odom.header.frame_id = self.odom_frame
        odom.child_frame_id = self.base_frame
        odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z = map(float, p)
        odom.pose.pose.orientation.x, odom.pose.pose.orientation.y, \
            odom.pose.pose.orientation.z, odom.pose.pose.orientation.w = map(float, q)
        odom.twist.twist.linear.x, odom.twist.twist.linear.y, odom.twist.twist.linear.z = map(float, result.vel)
        self.odom_pub.publish(odom)

        tf = TransformStamped()
        tf.header.stamp = stamp; tf.header.frame_id = self.odom_frame
        tf.child_frame_id = self.base_frame
        tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z = map(float, p)
        tf.transform.rotation.x, tf.transform.rotation.y, tf.transform.rotation.z, tf.transform.rotation.w = map(float, q)
        self.tf_bc.sendTransform(tf)

        if self.path_pub is not None:
            ps = PoseStamped(); ps.header = odom.header; ps.pose = odom.pose.pose
            self._path.poses.append(ps); self._path.header.stamp = stamp
            self.path_pub.publish(self._path)


def main(args=None):
    rclpy.init(args=args)
    node = LioNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd ros2_ws/src/kiss_imu_ros && KISS_IMU_SRC=<repo>/src python -m pytest test/test_node_config.py -v`
Expected: PASS（1 passed）

- [ ] **Step 5: 提交**

```bash
git add ros2_ws/src/kiss_imu_ros/kiss_imu_ros/lio_node.py ros2_ws/src/kiss_imu_ros/test/test_node_config.py
git commit -m "feat(ros): rclpy lio_node — IMU/points subs, odom+TF pubs, MT executor"
```

---

## Task 7: 离线回放验证 + README

**Files:**
- Create: `ros2_ws/src/kiss_imu_ros/test/test_replay_matches_offline.py`
- Create: `ros2_ws/src/kiss_imu_ros/README.md`

把现成的合成数据集 `data/synth/Synth01` 用 `SeqDataset` 逐窗口喂进 `OnlineEstimator.step()`，验证轨迹长度正确、位姿有限——把"算法连贯性"和 ROS 解耦。（精度 vs GT 的对比留到真机/带 GT 的 bag。）

- [ ] **Step 1: 写回放测试**

`test/test_replay_matches_offline.py`：
```python
import os
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pypose")
pytest.importorskip("small_gicp")

from kiss_imu_ros.paths import ensure_src_on_path
ensure_src_on_path()
from data.seq_dataset import SeqDataset  # noqa: E402
from kiss_imu_ros.online_estimator import OnlineEstimator, EstimatorConfig  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
SYNTH = os.path.join(REPO, 'data', 'synth')


@pytest.mark.skipif(not os.path.isdir(os.path.join(SYNTH, 'Synth01')),
                    reason="synth dataset not generated; run tools/gen_synth_dataset.py")
def test_replay_synth_produces_finite_trajectory():
    ds = SeqDataset(data_root=SYNTH, data_seq='Synth01', data_type='diter_os')
    cfg = EstimatorConfig(lo_model='small_gicp', device='cpu',
                          gravity=tuple(np.asarray(ds.gravity).reshape(-1).tolist()))
    est = OnlineEstimator(cfg)

    n = min(20, len(ds))
    poses = []
    for i in range(n):
        item = ds[i]
        accels = item['accels'].numpy()
        gyros = item['gyros'].numpy()
        imu_ts = item['imu_ts'].numpy()
        scan1 = item['scan1'].numpy()
        r = est.step(scan1, accels, gyros, imu_ts)
        poses.append(r.pose)
    poses = np.asarray(poses)
    assert poses.shape == (n, 7)
    assert np.all(np.isfinite(poses))
```

- [ ] **Step 2: 确保合成数据存在，跑测试**

Run:
```bash
cd <repo> && python tools/gen_synth_dataset.py --out data/synth/Synth01
cd ros2_ws/src/kiss_imu_ros && KISS_IMU_SRC=<repo>/src python -m pytest test/test_replay_matches_offline.py -v
```
Expected: PASS（1 passed）

- [ ] **Step 3: 写 README.md**

```markdown
# kiss_imu_ros — 实时 ROS2 LiDAR-惯性里程计前端（Phase 1）

基于 KISS-IMU 的流式 LIO 前端。Phase 1：raw IMU + ICP + 2 节点 PVGO，发布 odom + TF。

## 依赖
- ROS2（rclpy, sensor_msgs, nav_msgs, tf2_ros, sensor_msgs_py）
- Python: torch, pypose, numpy, small_gicp（或真机用 kiss_icp）
- 上游算法在 `<repo>/src`，通过 `KISS_IMU_SRC` 环境变量或默认相对路径定位。

## 测试（无需 ROS / 传感器 / GPU）
```bash
cd ros2_ws/src/kiss_imu_ros
KISS_IMU_SRC=<repo>/src python -m pytest test/ -v
```

## 构建与运行（ROS2）
```bash
cd ros2_ws
export KISS_IMU_SRC=<repo>/src
colcon build --packages-select kiss_imu_ros
source install/setup.bash
ros2 launch kiss_imu_ros lio.launch.py
```

## bag 回放
```bash
ros2 launch kiss_imu_ros lio.launch.py &
ros2 bag play your_dataset.db3 --remap /your_imu:=/imu /your_points:=/points
# 看 /odometry、TF odom->base_link，rviz2 加 Odometry/Path/TF
```

## 必改项
- `config/lio.yaml` 的 `R_I_L`/`T_I_L` 必须按实机 IMU→LiDAR 标定填写。
- 真机建议 `lo_model: kiss_icp`，`device: cuda:0`。
```

- [ ] **Step 4: 提交**

```bash
git add ros2_ws/src/kiss_imu_ros/test/test_replay_matches_offline.py ros2_ws/src/kiss_imu_ros/README.md
git commit -m "test(ros): synth replay smoke + package README"
```

---

## Task 8: 全量测试与收尾

- [ ] **Step 1: 跑全部单测**

Run: `cd ros2_ws/src/kiss_imu_ros && KISS_IMU_SRC=<repo>/src python -m pytest test/ -v`
Expected: 全部 PASS（small_gicp 缺失时相关用例 skip，不算失败）

- [ ] **Step 2: colcon 构建确认（若本机有 ROS2）**

Run: `cd ros2_ws && colcon build --packages-select kiss_imu_ros`
Expected: 构建成功；`ros2 launch kiss_imu_ros lio.launch.py` 能起节点并打印 `lio_node ready`。

- [ ] **Step 3: 提交收尾（如有 .gitignore / colcon 产物处理）**

```bash
# 忽略 colcon 产物
printf 'ros2_ws/build/\nros2_ws/install/\nros2_ws/log/\n' >> .gitignore
git add .gitignore
git commit -m "chore(ros): ignore colcon build artifacts"
```

---

## Self-Review（计划自查结果）

- **Spec 覆盖**：§2 包结构→Task0；§3 OnlineEstimator/可插拔 corrector→Task3/4；§4 数据流/IMU 缓冲→Task1/6；§5 ROS 接口→Task5/6；§6 实时线程隔离→Task6（MultiThreadedExecutor + 回调组）；§8 测试→Task1-4/7；§9 Phase 1 范围（raw+kiss_icp+最小窗口 PVGO）→全计划。**Phase 2（滑动窗口/降采样/高频 TF）、Phase 3（LearnedCorrector）不在本计划**，将各自单独成计划——符合"每个计划交付可运行软件"。
- **占位符扫描**：无 TBD/TODO；每个改码步骤都含完整代码。
- **类型一致性**：`build_window_sample`/`RawCorrector.correct` 的返回键与 `IMUNet.forward`（`src/models/imu_net.py:136-143`）一致；`OnlineEstimator.step` 的调用签名与 `inference.py:107-149` 一致；`params_to_cfg`/`EstimatorConfig` 字段在 Task4/6 一致。
- **已知风险（实现时核对）**：(1) `small_gicp` 在随机合成点云上配准可能不收敛，故 Task4/7 仅做契约/有限性断言，不断言精度；(2) 真机外参必须替换默认单位阵；(3) `kiss_icp` 节点内初始化需在真实点云上验证；(4) `device='cuda:0'` 在无 GPU 环境测试时用 `cpu`。
