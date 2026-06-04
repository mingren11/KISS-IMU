# kiss_imu_ros 使用流程与注意事项（中文）

本文档讲清楚怎么用基于 KISS-IMU 的实时 LiDAR-惯性里程计（LIO）**前端**：有哪几条路径、每条怎么跑、以及真机部署的注意事项。

> 这是**里程计前端**，不是完整 SLAM —— 没有回环检测、全局优化、重定位。

相关文档：
- 包功能/参数速查：[`README.md`](README.md)
- 项目母设计 / 各阶段 spec：`docs/superpowers/specs/`、实现计划 `docs/superpowers/plans/`
- 仓库整体与合成数据 demo：`../../RUNNING.md`、`../../PROJECT_OVERVIEW_zh.md`

---

## 0. 三条使用路径，先选对

| 路径 | 用途 | 需要 ROS2？ | 需要 GPU？ |
| --- | --- | --- | --- |
| **A. 合成数据冒烟** | 验证算法链路能跑（无需任何真实数据/传感器） | 否 | 否 |
| **B. Livox 离线 bag 回放** | 用你录的 **Mid-360 bag**（CustomMsg）离线验证前端、出轨迹图 | 否 | 否 |
| **C. 真机实时节点** | 机器人上实时跑，发布 Odometry + TF | **是** | 推荐有 |

- 你录的 `data/mid360_1`（Livox CustomMsg）→ 走**路径 B**（节点不直接吃 CustomMsg，见 §4.1）。
- Velodyne，或 Mid-360 切到 PointCloud2 输出 → 可走**路径 C**。

---

## 1. 环境准备

### 1.1 离线路径（A / B）— conda 环境
核心依赖：`torch`、`pypose`、`numpy`、`scikit-learn`、`matplotlib`。
LiDAR 配准后端：离线测试用 `small_gicp`（pip 可装）。路径 B 还需 `rosbags`。

```bash
# 以本仓库使用的 air-io 环境为例
~/anaconda3/envs/air-io/bin/pip install small_gicp rosbags
```

### 1.2 真机路径（C）— ROS2
机器人上需要 ROS2（Humble+）、`rclpy`、`sensor_msgs`、`nav_msgs`、`tf2_ros`、`sensor_msgs_py`，以及 Python 侧 `torch`、`pypose`，配准后端推荐 `kiss-icp`（`pip install kiss-icp`）。

> 上游算法代码在 `<repo>/src`，通过环境变量 `KISS_IMU_SRC` 定位（默认从包相对路径推断）。运行任何导入估计器的命令时，建议显式 `export KISS_IMU_SRC=<repo>/src`。

---

## 2. 路径 A：合成数据冒烟（最快验证链路）

```bash
cd <repo>
# 1) 生成合成序列（一次即可）
python tools/gen_synth_dataset.py --out data/synth/Synth01
# 2) 跑包内单元测试（含合成数据回放，纯 CPU、无 ROS）
cd ros2_ws/src/kiss_imu_ros
KISS_IMU_SRC=<repo>/src python -m pytest test/ -v
```
期望：`28 passed, 1 skipped`（跳过的是需要 rclpy 的节点测试）。这证明：IMU 缓冲、点云转换、去畸变时间归一化、估计器（IMU+ICP+PVGO）、发散保护、Livox 转换都正常。

---

## 3. 路径 B：Livox（Mid-360）离线 bag 回放

让你的 Mid-360 bag 在**无 ROS、无 Livox 驱动**的情况下跑通整条前端，输出俯视轨迹图。

```bash
cd <repo>
KISS_IMU_SRC=$PWD/src python tools/replay_livox_bag.py \
  --bag data/mid360_1 \
  --lo-model small_gicp \
  --device cpu \
  --out results/livox_mid360_1
# 看结果：results/livox_mid360_1/trajectory.png  + trajectory.npz
```

常用参数：

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--bag` | （必填） | rosbag2 目录（含 `.db3` + `metadata.yaml`） |
| `--imu-topic` | `/livox/imu` | IMU 话题（sensor_msgs/Imu） |
| `--lidar-topic` | `/livox/lidar` | LiDAR 话题（livox_ros_driver2/CustomMsg） |
| `--acc-scale` | `9.80665` | IMU 加速度缩放：**Livox 报的是 g**，乘 9.80665 转 m/s²。若你的数据已是 m/s² 用 `1.0` |
| `--lo-model` | `small_gicp` | 配准后端（`small_gicp` 离线最省事；`kiss_icp` 会用 per-point 时间做去畸变） |
| `--device` | `cpu` | `cpu` 或 `cuda:0` |
| `--max-frames` | `0`（全部） | 限制处理帧数，调试用 |
| `--out` | `results/livox_replay` | 输出目录 |

注意：
- 这类 bag **没有 GT 位姿**，验证是**定性**的（看 `trajectory.png` 轨迹是否合理、是否漂移）。
- per-point `offset_time` 已用于去畸变（归一化到 [0,1]）。
- 想看明显轨迹，请用**有运动**的序列（静止/缓动序列轨迹会很短）。

---

## 4. 路径 C：真机实时 ROS2 节点

### 4.1 先确认你的雷达消息类型
节点订阅 **`sensor_msgs/PointCloud2`**，不订阅 Livox `CustomMsg`。

```bash
ros2 topic info /your_lidar_topic        # 看 Type
```
- 是 `sensor_msgs/msg/PointCloud2` → 可直接用（Velodyne，或 Mid-360 把 livox 驱动 `xfer_format` 设为 PointCloud2）。
- 是 `livox_ros_driver2/msg/CustomMsg` → 节点收不到。两个办法：把驱动切到 PointCloud2 输出；或先用**路径 B** 离线验证（实时 CustomMsg 节点适配是后续工作）。

### 4.2 配置 `config/lio.yaml`（**部署前必改**）
```yaml
/lio_node:
  ros__parameters:
    imu_topic: "/imu"
    points_topic: "/points"
    lo_model: "kiss_icp"          # 真机推荐
    device: "cuda:0"              # 无 GPU 用 "cpu"
    R_I_L: [...]                  # ★ IMU→LiDAR 旋转(行主序3x3)，必须按实机标定
    T_I_L: [...]                  # ★ IMU→LiDAR 平移，必须按实机标定
    gravity: [0.0, 0.0, 9.81]
    time_field: ""                # 去畸变用：Velodyne→"time"，Mid-360(PointCloud2)→"timestamp"；空=不去畸变
    voxel_size: 0.0               # 节点级降采样，默认关（见 §5）
    max_step_ms: 0.0              # >0 时 step() 超时打 warn；0=只丢帧不警告
```

### 4.3 构建并运行
```bash
cd <repo>/ros2_ws
export KISS_IMU_SRC=<repo>/src
colcon build --packages-select kiss_imu_ros
source install/setup.bash
ros2 launch kiss_imu_ros lio.launch.py
```
用 bag 回放驱动（话题名不一致时 remap）：
```bash
ros2 bag play your.db3 --remap /your_imu:=/imu /your_points:=/points
```
在 rviz2 里加 Odometry / Path / TF（`odom`→`base_link`）查看。

---

## 5. 注意事项（踩过的坑都在这）

1. **IMU 加速度单位**：Livox 内置 IMU 报 **g**，不是 m/s²。离线工具用 `--acc-scale 9.80665` 已处理；真机若用 Livox IMU，喂给节点前也要确认单位是 m/s²（否则重力/积分全错）。
2. **Mid-360 是 CustomMsg**：节点只吃 PointCloud2。CustomMsg 走离线工具，或改驱动输出格式。
3. **外参 `R_I_L`/`T_I_L` 必须标定**：默认是单位阵/零，真机不改会导致 LiDAR 与 IMU 坐标错位、轨迹发散。
4. **`time_field`（去畸变）**：不设则 `scan1_ts` 零填充 = **去畸变关闭**。`kiss_icp` 后端才用它；`small_gicp` 忽略。旋转 LiDAR（Velodyne）务必设对。
5. **QoS 匹配**：节点订阅用 `sensor_data`（BestEffort）。若上游发布是 RELIABLE，ROS2 会静默不匹配、节点收不到消息且无报错——对齐两端 QoS。
6. **device**：`cuda:0`（默认）需可用 GPU；无 GPU 必须显式 `device: cpu`（并对离线工具用 `--device cpu`）。
7. **`voxel_size` 默认关（0.0）**：ICP 后端内部已各自体素化，节点再降一次会双重降采样、掉精度。仅在点数过大需限流时才开；开启时 per-point 时间会按相同索引对齐下采样。
8. **实时性**：点云回调用非阻塞门控，上一帧 `step()` 没跑完就**丢弃**新帧（保最新、不积压延迟）。设 `max_step_ms>0` 可在超时打 warn 并报累计丢帧数。
9. **发散保护**：某一帧 PVGO 解出非有限值时，自动回退到该帧的 IMU 预积分位姿，避免污染后续所有状态。
10. **当前能力边界**：2 节点 PVGO（非滑动窗口）、Odometry 暂不输出协方差、无回环/全局优化、无学习式 IMU 修正（`RawCorrector` 透传）。这些是后续阶段。

---

## 6. 故障排查

| 现象 | 可能原因 / 处理 |
| --- | --- |
| 节点收不到点云/IMU | 话题名不对（remap）；或 QoS 不匹配（§5.5） |
| 轨迹很快发散/乱飞 | 外参没标定（§5.3）；IMU 加速度单位错（§5.1）；重力方向不对 |
| `cuda` 报错/显存不足 | 改 `device: cpu` / `--device cpu` |
| 旋转时轨迹有畸变 | 没设 `time_field`，去畸变没开（§5.4） |
| 实时延迟越积越大 | 正常会丢帧；若帧率过低，降点云密度或换 GPU；设 `max_step_ms` 观察耗时 |
| 离线工具 `No module named rosbags` | `pip install rosbags` |
| pytest 节点测试 skipped | 正常——该测试需 `rclpy`，离线 conda 环境没有 |
