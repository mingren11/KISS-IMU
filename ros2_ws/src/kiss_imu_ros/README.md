# kiss_imu_ros — real-time ROS2 LiDAR-inertial odometry front-end (Phase 1)

Streaming LIO front-end built on KISS-IMU. Phase 1: raw IMU + ICP + 2-node PVGO, publishes odom + TF. No loop closure, no global optimization (front-end only).

## Dependencies
- ROS2 (rclpy, sensor_msgs, nav_msgs, tf2_ros, sensor_msgs_py)
- Python: torch, pypose, numpy, small_gicp (or kiss_icp on the real robot)
- The upstream algorithm code lives in `<repo>/src`, located via the `KISS_IMU_SRC` env var (default: a relative path from the package).

## Tests (no ROS / no sensor / no GPU needed)
```bash
cd ros2_ws/src/kiss_imu_ros
KISS_IMU_SRC=/abs/path/to/KISS-IMU/src python -m pytest test/ -v
```
The replay test additionally needs the synthetic dataset:
```bash
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

## Must-configure
- `config/lio.yaml` `R_I_L` / `T_I_L` MUST be set from your real IMU->LiDAR calibration (defaults are identity).
- On the real robot use `lo_model: kiss_icp`, `device: cuda:0`.

## QoS note
Both subscriptions use `qos_profile_sensor_data` (BestEffort). If your IMU/LiDAR publisher uses RELIABLE QoS, ROS2 will silently fail to match and the node receives nothing — align the publisher QoS or adjust the subscription QoS accordingly.

## Scope (Phase 1)
Front-end odometry only. Deferred to later phases: learned IMU correction (LearnedCorrector), sliding-window PVGO, input voxel downsampling, high-rate IMU TF, and a NaN/finite guard on the optimized anchor pose (a divergent solve currently propagates into subsequent steps).
