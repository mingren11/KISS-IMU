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
