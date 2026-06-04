# kiss_imu_ros — real-time ROS2 LiDAR-inertial odometry front-end

Streaming LIO front-end built on KISS-IMU. Raw IMU + ICP + 2-node PVGO, publishes odom + TF. No loop closure, no global optimization (front-end only). Phase 2a adds per-point deskew, optional node-level voxel downsampling, and a frame-drop gate with latency monitoring.

## Dependencies
- ROS2 (rclpy, sensor_msgs, nav_msgs, tf2_ros, sensor_msgs_py)
- Python: torch, pypose, numpy, small_gicp (`pip install small_gicp`). For the real-robot path (`lo_model: kiss_icp`) also install kiss-icp separately: `pip install kiss-icp`.
- The upstream algorithm code lives in `<repo>/src`, located via the `KISS_IMU_SRC` env var (default: a relative path from the package).

## Tests (no ROS / no sensor / no GPU needed)
```bash
cd ros2_ws/src/kiss_imu_ros
KISS_IMU_SRC=/abs/path/to/KISS-IMU/src python -m pytest test/ -v
```
The replay test additionally needs the synthetic dataset:
```bash
# from the repo root (<repo>/), not from ros2_ws/...
python tools/gen_synth_dataset.py --out data/synth/Synth01
```

## Build & run (ROS2)
```bash
cd ros2_ws
export KISS_IMU_SRC=/abs/path/to/KISS-IMU/src
colcon build --packages-select kiss_imu_ros
source install/setup.bash
ros2 launch kiss_imu_ros lio.launch.py
```

## Bag replay
```bash
ros2 launch kiss_imu_ros lio.launch.py &
ros2 bag play your_dataset.db3 --remap /your_imu:=/imu /your_points:=/points
# inspect /odometry, TF odom->base_link; in rviz2 add Odometry / Path / TF
```

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

## Must-configure
- `config/lio.yaml` `R_I_L` / `T_I_L` MUST be set from your real IMU->LiDAR calibration (defaults are identity).
- On the real robot use `lo_model: kiss_icp`, `device: cuda:0`.
- For deskew, set `time_field` to your LiDAR's per-point time field: Velodyne → `time`, Livox Mid-360 (PointCloud2 mode) → `timestamp` (empty disables deskew). **Mid-360 must publish `sensor_msgs/PointCloud2`, not `livox_ros_driver2/msg/CustomMsg`** — set the Livox driver `xfer_format` to PointCloud2 (or convert), otherwise the node receives nothing.

## Real-time
- The scan callback uses a non-blocking gate: if a `step()` is still running when a new scan arrives, the new scan is **dropped** (keeps latest, bounds latency). Set `max_step_ms` > 0 to log a warning + cumulative drop count when a step is slow.
- `voxel_size` node-level downsampling defaults to OFF (`0.0`). The ICP backends voxelize internally; enabling node-level downsampling double-downsamples and can hurt accuracy. Only enable it to cap point count for transport/overlap cost; when enabled, per-point deskew times are downsampled with the same indices to stay aligned.

## QoS note
Both subscriptions use `qos_profile_sensor_data` (BestEffort). If your IMU/LiDAR publisher uses RELIABLE QoS, ROS2 will silently fail to match and the node receives nothing — align the publisher QoS or adjust the subscription QoS accordingly.

## Scope
Front-end odometry only. Phase 2a (deskew, optional downsample, frame-drop/latency) is implemented. Still deferred to later phases: sliding-window PVGO, Odometry covariance, high-rate IMU TF, and learned IMU correction (LearnedCorrector). NaN/finite guard on the optimized anchor pose is implemented (divergent solves fall back to the IMU-integrated node).
