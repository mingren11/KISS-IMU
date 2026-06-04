# 设计：kiss_imu_ros Phase 2a — 真机 LiDAR 正确性 + 实时安全

- 日期：2026-06-04
- 状态：已批准（待实现计划）
- 依赖：Phase 1 已合并（`ros2_ws/src/kiss_imu_ros/`）。母设计：`2026-06-04-realtime-ros2-lio-frontend-design.md`

## 1. 背景与目标

Phase 1 交付了能跑的实时 LIO 前端，但有两个真机短板：

1. **去畸变实际是关闭的。** `build_window_sample` 把 `scan1_ts` 零填充，而推荐的真机后端 `kiss_icp` 的 `deskew_scan(curr_scan, curr_ts, last_delta)` 依赖 per-point 时间戳做运动补偿。零填充 = 无补偿，旋转/运动中的 LiDAR 会有畸变误差。
2. **无实时上界。** 点云回调串行，但 `step()` 慢于 scan 周期时回调排队积压，里程计延迟会无限累积。

目标硬件：机器人有 **Livox Mid-360**（非重复扫描，每点带 offset 时间）和 **Velodyne**（`time` 字段=相对扫描起始秒，带 `ring`）。本阶段让 `kiss_icp` 在这两款真实 LiDAR 上**跑对**，并给实时性加上界。

**范围（YAGNI）**：(a) per-point 去畸变时间戳；(b) 可选点云降采样（默认关）；(c) 延迟监控 + 丢帧门控。**不含**滑动窗口 PVGO、Odometry 协方差、高频 IMU TF（属 Phase 2b/2c），不含学习模型（Phase 3）。

## 2. per-point 去畸变时间戳

**思路：可配置字段名 + 归一化，不为每款雷达写专用解析。** 按配置的 `time_field` 从 PointCloud2 取 per-point 时间数组，统一归一化到 `[0,1]`（scan 内比例），即 kiss_icp 期望格式。一份代码两款雷达通用；无时间字段时回退零填充（等于关去畸变，保持 Phase 1 行为）。

归一化对所有单位都成立：`t_norm = (t - t.min()) / (t.max() - t.min() + eps)`，适配 Velodyne 相对秒、Livox offset ns、绝对时间戳；退化（全相等）回退零填充。

**改动落点：**
- `pointcloud2.py`：`xyz_from_pointcloud2(msg, field_names=('x','y','z'), time_field=None)` → 返回 `(xyz (N,3) float64, t_norm (N,) float64 或 None)`。`time_field` 为 None 或字段缺失 → 返回 `(xyz, None)`。读取后用与 xyz 相同的 finite 掩码对齐过滤。
- 新增纯函数 `normalize_per_point_times(t) -> np.ndarray`：实现上面的归一化，退化时返回全零。
- `imu_corrector.build_window_sample(..., scan1_ts=None)`：已支持传入 per-point 时间（Phase 1 已是 None→零填充）；2a 把真实归一化时间传进来，零填充逻辑保留作回退。
- `lio_node`：参数 `time_field`（默认空字符串=不去畸变；预设值：Velodyne→`"time"`，Livox→`"timestamp"`）。`on_points` 解析时取 `t_norm` 并经 `scan1_ts` 传入 `estimator.step()`。
- `OnlineEstimator.step(scan_xyz, accels, gyros, imu_ts, scan1_ts=None)`：新增可选 `scan1_ts` 参数，透传给 `build_window_sample`。默认 None 保持向后兼容（现有测试不变）。
- `lio.yaml`：新增 `time_field: ""`，注释说明两款雷达的预设值。

**仅影响 kiss_icp 路径**：`small_gicp` 后端不读 `scan1_ts`，所以 Phase 1 的 small_gicp 测试与行为不变。

**测试（纯 numpy，无 ROS）：**
- `normalize_per_point_times`：Velodyne 风格（相对秒，如 `linspace(0,0.1,N)`）和 Livox 风格（offset ns，大整数）输入 → 断言结果 ∈[0,1]、首≈0、末≈1；全相等输入 → 全零。
- `xyz_from_arrays` 路径已测；为 time 提取补一个结构化数组测试（含/不含 time_field 两种）。

## 3. 可选点云降采样

- 现状：`voxel_size` 参数已声明但未接线；`pointcloud2.voxel_downsample` 已实现且已测。
- 设计：`lio_node.on_points` 解析点云后、传入 `step()` 前，**可选**体素降采样。
- **默认关闭（`voxel_size <= 0`）。** 理由：kiss_icp / small_gicp 后端内部已各自体素化，节点再降一次会双重降采样、掉精度。节点级降采样只在点数过大、传输/overlap 计算吃紧时手动开。
- **顺序坑**：降采样会破坏点与 per-point 时间的对应。因此本阶段节点级降采样默认关，真正的体素化交给后端内部完成；若用户显式开启节点级降采样，则**同时对 `t_norm` 用相同的保留索引下采样**以保持对齐（`voxel_downsample` 需返回保留索引的变体，或在节点里用相同 grid 逻辑同步抽取）。
- yaml/README 写清这层交互。

**测试**：`voxel_size <= 0` 时点云原样通过（断言形状不变）；开启时点数减少且 `t_norm` 长度与点数一致。

## 4. 延迟监控 + 丢帧门控

- 现状：点云回调在独立互斥回调组串行，但慢 `step()` 会让回调排队、延迟累积。
- 设计：引入轻量 `FrameGate`（纯 Python，无 ROS）：
  - `try_enter() -> bool`：非阻塞 `threading.Lock.acquire(blocking=False)`；成功返回 True，失败（上一帧仍在跑）返回 False 并 `dropped += 1`。
  - `exit()`：释放锁。
  - `dropped` 计数可读。
- `on_points`：`if not gate.try_enter(): return`（丢弃积压帧，保最新）。`try/finally` 保证 `exit()`。
- 每帧用 `time.monotonic()` 量 `step()` 耗时；超过 `max_step_ms`（参数，0=不警告）则 `WARN` 日志并周期性报告累计丢帧数。
- 效果：延迟有上界、不累积；代价是高负载时有效帧率下降（里程计可接受）。
- 参数：`max_step_ms: 0.0`（0=只丢帧不警告）。

**测试（无 ROS）**：`FrameGate` 单测——`try_enter()` 成功后再次 `try_enter()` 返回 False 且 `dropped` +1；`exit()` 后再 `try_enter()` 成功。

## 5. 复用 vs 新代码

| 复用（0 改） | 修改 | 新增 |
| --- | --- | --- |
| `ring_buffer`, `RawCorrector`, `optimize`, `LOModule`, `IMUIntegrator` | `pointcloud2.py`(+time)、`imu_corrector`(透传 scan1_ts，已支持)、`online_estimator.step`(+scan1_ts 参数)、`lio_node`(time_field/voxel/gate 接线)、`lio.yaml` | `pointcloud2.normalize_per_point_times`、`frame_gate.py`(`FrameGate`)、对应单测 |

`OnlineEstimator.step` 新增的 `scan1_ts` 参数默认 None → Phase 1 测试与调用全部向后兼容。

## 6. 测试策略

- 全部新逻辑（归一化、time 提取、FrameGate、降采样对齐）做成纯函数/小类，无 ROS、CPU 可测。
- 现有 Phase 1 测试必须继续全绿（向后兼容）。
- 端到端验证（真机/bag）不在单测内：用带 `time` 字段的 Velodyne bag 或 Livox bag 回放，目视 kiss_icp 去畸变后轨迹更干净；丢帧在人为降算力时计数上升。
- 测试环境：conda `air-io`（torch/pypose/small_gicp）；kiss_icp 真机路径需在机器人 ROS2 环境验证（air-io 无 kiss_icp/rclpy）。

## 7. 风险与实现时核对

1. **kiss_icp 期望的时间戳语义**需对实际安装版本核对：多数版本 `deskew_scan` 接受 [0,1] 归一化时间；若该版本要求秒/相对，实现时调整归一化目标（保留 `normalize_per_point_times` 作为单一改动点）。
2. **Livox Mid-360 的 PointCloud2 字段名/单位**随 `livox_ros_driver2` 版本而变（可能是 `timestamp` 绝对值、或需用 CustomMsg 的 `offset_time`）；`time_field` 可配 + 归一化吸收单位差异，但字段名需对实际 topic 核对（`ros2 topic echo --field` 看 PointCloud2 fields）。Mid-360 若以 CustomMsg 而非 PointCloud2 发布，需先在驱动侧切到 PointCloud2 输出（本阶段假设 PointCloud2 输入）。
3. **降采样破坏时间对齐**：默认关已规避；开启路径必须同步下采样 `t_norm`。
4. `max_step_ms` 默认 0（仅丢帧不警告），避免无标定时日志噪声。
