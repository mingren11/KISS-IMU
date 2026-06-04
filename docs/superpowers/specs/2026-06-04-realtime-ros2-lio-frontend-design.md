# 设计：基于 KISS-IMU 的实时 ROS2 LiDAR-惯性里程计前端

- 日期：2026-06-04
- 状态：已批准（待实现计划）
- 适用：ROS2 + GPU + LiDAR/IMU 真机；只做里程计前端，**不含回环/全局优化/重定位**

## 1. 背景与目标

KISS-IMU 本质是一套 **LiDAR-惯性里程计（LIO）的训练 + 离线推理管线**，不是完整 SLAM。
它已经具备 SLAM 前端的大部分零件：学习式 IMU 去噪（`IMUNet`）+ IMU 预积分
（`IMUIntegrator`）+ LiDAR ICP 配准（`LOModule`）+ 局部位姿-速度图优化
（`pvgo.optimize`）。

目标：把这套前端**改造成能在真机上实时运行的 ROS2 里程计节点**。核心难点不在算法，
而在「离线批处理 → 流式有状态增量估计 + 实时 + 鲁棒」。

明确**不做**：回环检测、全局位姿图优化、重定位、地图持久化、相机/多传感器融合。

### 现状关键事实（实现依据）

- `src/inference.py:99-159` 的循环体已经在用 `anchor_pose` / `anchor_vel`
  在窗口间传递状态。**该循环体即在线 `step()` 的蓝本**：把 batch 设为 1、把循环体抽成
  方法即可。
- `IMUNet` 每个窗口独立前向（GRU 不跨窗口），在线无需维护跨帧隐状态。
- `pvgo.optimize` 目前是**整条轨迹一次性**优化（需改造成滑动窗口）。
- `LOModule` 已封装 `kiss_icp` / `small_gicp` 等后端，`kiss_icp` 自带去畸变 + 局部地图，
  适合流式；`small_gicp` 支持 submap。
- 重力、外参 `R_I_L` / `T_I_L`、各数据集列约定见 `src/data/seq_dataset.py` 的 `DATASET_CFG`。

## 2. 架构

两层结构，新建独立 ROS2 包，**不改 `src/` 主体**（仅加极小适配器），便于跟上游同步：

```
ros2_ws/src/kiss_imu_ros/
├── kiss_imu_ros/
│   ├── online_estimator.py   # 纯 Python，无 ROS 依赖 —— 估计器核心
│   ├── lio_node.py           # rclpy 节点 —— 只管 ROS I/O
│   ├── imu_corrector.py      # 可插拔 IMU 级：LearnedCorrector / RawCorrector
│   ├── pointcloud2.py        # PointCloud2 <-> Nx3 numpy 转换
│   └── ring_buffer.py        # 线程安全 IMU 缓冲
├── config/                   # extrinsics、gravity、topics、window_size 等 yaml
├── launch/                   # lio.launch.py
├── test/                     # 离线回放单测（无 ROS / 无传感器）
├── package.xml
└── setup.py                  # ament_python
```

**关键边界：`OnlineEstimator` 完全不依赖 ROS。** 吃 numpy 点云 + IMU 数组，吐位姿。
好处：能脱离机器人、用录制序列离线单测；最大化复用 `inference.py` 已验证的逻辑。

## 3. OnlineEstimator（核心）

由 `src/inference.py:99-159` 循环体重构而来：

```python
class OnlineEstimator:
    def __init__(self, cfg): ...   # 建 imu_corrector / IMUIntegrator / LOModule / 滑窗 PVGO
    def step(self, scan_xyz, imu_window) -> OdomResult:
        # 1. imu_corrector(imu_window)        -> 校正后 acc/gyro/cov（可插拔）
        # 2. IMUIntegrator.integrate x2       -> imu_nodes / 相对运动 + 协方差
        # 3. lo_model(scan, anchor)           -> ICP 相对位姿 + overlap
        # 4. optimize(滑动窗口 IMU+ICP 因子)  -> 优化后位姿/速度
        # 5. 推进 anchor_pose/anchor_vel，滑窗出队
        return OdomResult(pose, vel, cov, overlap)
```

**携带状态**：`anchor_pose`(SE3)、`anchor_vel`(3)、最近 W 个节点的滑动窗口
（位姿 + IMU 因子 + ICP 因子）、`LOModule` 内部 submap。

**可插拔 IMU 级**（`imu_corrector.py`）：
- `LearnedCorrector(ckpt)`：加载 `IMUNet`，输出校正后的 acc/gyro/cov。
- `RawCorrector`：恒等透传（等价 `raw_pvgo` baseline），输出**相同字典结构**。
- 两者下游完全一致 → 学习模型可后插。

**滑动窗口 PVGO**：保留最近 W≈5~10 个节点及其 IMU/ICP between-factor，每步重优化；
最老节点作为 gauge 锚固定（`pvgo.optimize` 本就把首节点锚定，顺手）。输出最新节点的
位姿/速度作为里程计输出。

## 4. 数据流与同步

```
/imu (≈200Hz) ──► 线程安全环形缓冲 (t, acc, gyro)
                        │
/points (≈10Hz, stamp t_k) ──► 取出 (t_{k-1}, t_k] 区间 IMU + PointCloud2→Nx3
                        ▼
              构造 SeqDataset 契约窗口 dict
              (accels, gyros, imu_ts, imu_dts, scan0=上帧, scan1=本帧, valid_length)
                        ▼
                 estimator.step()  ──►  OdomResult
                        ▼
        发布 nav_msgs/Odometry + TF(odom→base_link)
```

- 全程用**传感器时间戳**（`header.stamp`），不用墙钟。
- 第一帧点云：bootstrap（状态置初值、存点云、不出里程计）。
- IMU/LiDAR 时间错位按 scan 时间戳切区间；外参 / 重力从 config 读。

## 5. ROS2 接口

**订阅**
- `~/imu` — `sensor_msgs/Imu`，QoS `sensor_data`(BestEffort)，~200Hz
- `~/points` — `sensor_msgs/PointCloud2`，~10Hz

**发布**
- `~/odometry` — `nav_msgs/Odometry`（位姿 + 速度 + 协方差，来自 PVGO/overlap）
- TF：`odom → base_link`
- `~/local_map` — `PointCloud2`（可选，submap 可视化）
- `~/path` — `nav_msgs/Path`（可选，rviz）

**参数（yaml + ros2 param）**：`ckpt_path`（或 `"raw"` 走 RawCorrector）、
`lo_model`（默认 `kiss_icp`）、`use_submap`、`lm_weight`、`window_size`、`device`(`cuda:0`)、
`R_I_L` / `T_I_L`、`gravity`、`voxel_size`、`frame_id` / `child_frame_id`、输入 topic 名。

## 6. 复用 vs 新代码

| 直接复用（0 改动） | 新写 | 极小适配 |
|---|---|---|
| `IMUNet`、`IMUIntegrator`、`LOModule`、`pvgo.optimize`、`overlap_score` | `online_estimator.py`、`lio_node.py`、`imu_corrector.py`、点云/IMU 缓冲工具、launch/config、test | 用 `collate_fn` 把单帧窗口包成 batch-of-1 喂 `IMUNet`；滑动窗口版 `optimize` 包装 |

新包放 `ros2_ws/src/kiss_imu_ros/`，`src/` 主体不动。

## 7. 实时性策略（ROS2 + GPU）

- `IMUNet`（Conv1d+GRU，很小）跑 GPU，每窗口前向可忽略。
- **瓶颈是 ICP** → 输入点云体素降采样 + 限制点数；`kiss_icp` 流式友好。
- **线程隔离**：`MultiThreadedExecutor` + 独立 callback group。IMU 回调只入缓冲（轻）；
  重活 `step()` 在点云回调/worker 跑，不阻塞 IMU 进流。
- 目标：`step()` < 一个 scan 周期（10Hz→100ms）；加延迟监控，积压时跳帧。
- （可选，Phase 2）两帧之间用纯 IMU 预积分发高频 TF，每来一帧 scan 校正一次
  （类 LIO-SAM imuPreintegration），让 TF 在 IMU 频率上平滑。

## 8. 测试策略

`OnlineEstimator` 无 ROS → 算法正确性与 ROS 集成解耦：

1. **离线回放单测（先写，TDD）**：把序列按窗口逐个喂进 `estimator.step()`，轨迹与
   `inference.py` 批处理结果对比（窗口等价时应一致到容差内）。
2. **合成数据冒烟**：现成 `data/synth/Synth01` 作首个 `step()` 测试输入（纯 CPU 可过）。
3. **端到端 bag 回放**：`ros2 bag` 录段（或带 GT 数据集转 bag），跑节点看 odom 漂移 vs GT。
4. CI 至少跑 1+2（无传感器、无 GPU）。

## 9. 分期与范围（YAGNI）

- **Phase 1（让机器人动起来）**：`RawCorrector` + `kiss_icp` + 最小窗口 PVGO + ROS2 节点
  发 odom/TF，合成数据 + bag 测通。
- **Phase 2（实时 + 精度）**：滑动窗口 PVGO（W 可调）、降采样/实时调优、高频 IMU TF。
- **Phase 3（接学习模型）**：用机器人采集数据训 `IMUNet`，换 `LearnedCorrector`——纯增量升级。
- **不做**：回环、全局 PGO、重定位、地图持久化、相机/多传感器。

## 10. 风险与待实现时核对

- `pvgo.optimize` 滑动窗口化的数值稳定性（gauge 锚、协方差权重尺度）需实测。
- `IMUIntegrator.integrate` / `IMUNet.forward` 在 batch-of-1 下的张量维度需核对
  （复用 `collate_fn` 保证形状一致）。
- `kiss_icp` 在 ROS2 节点进程内的初始化与外参约定需与 `LOModule` 既有用法对齐。
- 真机 IMU/LiDAR 外参与 `gravity` 必须按实机标定填入 config，不能沿用 `DATASET_CFG` 默认值。
- `PROJECT_OVERVIEW_zh.md` §10 列出的若干上游实现细节（如 `evaluate.py` 返回值、
  `raw_pvgo.py` 接口）不在本前端路径上，但复用相关模块时需留意。
