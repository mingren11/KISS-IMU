"""Generate a tiny, physically-consistent synthetic sequence in the DiTer-OS
layout so the KISS-IMU pipeline can be exercised end-to-end without the real
(unreleased) datasets.

The trajectory is planar (yaw-only) and built from several distinct motion
regimes — slow/fast straights and gentle/sharp turns — so the GMM motion-regime
clustering has more than one component to find. IMU readings are derived from
the trajectory under the pypose convention ``a_world = R @ acc - g`` (verified
against pypose 0.6.8), i.e. ``acc_body = R^T (a_world + g)``, so that raw
integration already approximates the ground truth and the network only has to
learn small corrections.

Usage:
    python tools/gen_synth_dataset.py --out <data_root>/<seq_name>
"""

import os
import argparse
import numpy as np


G = 9.81
IMU_HZ = 200.0
SCAN_HZ = 10.0
GRAVITY = np.array([0.0, 0.0, G])

LIDAR_DTYPE = np.dtype([("x", "<f8"), ("y", "<f8"), ("z", "<f8"), ("intensity", "<f8")])


def _smooth(x, k=21):
    """Centered moving average to avoid acceleration spikes at regime changes."""
    ker = np.ones(k) / k
    return np.convolve(x, ker, mode="same")


def build_profiles(t):
    """Per-sample forward speed v(t) and yaw-rate w(t) over several regimes."""
    v = np.zeros_like(t)
    w = np.zeros_like(t)
    # (start_s, end_s, speed, yaw_rate)
    regimes = [
        (0.0, 3.0, 0.5, 0.0),    # slow straight
        (3.0, 6.0, 2.0, 0.0),    # fast straight
        (6.0, 9.0, 1.0, 0.5),    # gentle turn
        (9.0, 12.0, 0.4, 1.1),   # slow sharp turn
        (12.0, 1e9, 1.5, 0.25),  # medium sweeping turn
    ]
    for a, b, vv, ww in regimes:
        m = (t >= a) & (t < b)
        v[m] = vv
        w[m] = ww
    return _smooth(v), _smooth(w)


def quat_from_yaw(yaw):
    """[qx,qy,qz,qw] for a rotation of `yaw` about +z."""
    return np.stack([np.zeros_like(yaw), np.zeros_like(yaw),
                     np.sin(yaw / 2.0), np.cos(yaw / 2.0)], axis=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output dir <data_root>/<seq_name>")
    ap.add_argument("--duration", type=float, default=15.0, help="seconds")
    ap.add_argument("--acc-bias", type=float, default=0.03, help="constant accel bias [m/s^2]")
    ap.add_argument("--gyr-bias", type=float, default=0.005, help="constant gyro bias [rad/s]")
    ap.add_argument("--acc-noise", type=float, default=0.02)
    ap.add_argument("--gyr-noise", type=float, default=0.002)
    ap.add_argument("--n-points", type=int, default=1200, help="points per LiDAR scan")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    dt = 1.0 / IMU_HZ
    scan_ts = np.arange(0.0, args.duration, 1.0 / SCAN_HZ)
    n_scan = len(scan_ts)
    # IMU timeline covers slightly past the last scan so the final window is full.
    t = np.arange(0.0, scan_ts[-1] + 2.0 / SCAN_HZ, dt)
    N = len(t)

    v, w = build_profiles(t)
    yaw = np.cumsum(w) * dt
    heading = np.stack([np.cos(yaw), np.sin(yaw), np.zeros_like(yaw)], axis=1)
    left = np.stack([-np.sin(yaw), np.cos(yaw), np.zeros_like(yaw)], axis=1)

    # world velocity / position
    p_dot = v[:, None] * heading
    pos = np.cumsum(p_dot, axis=0) * dt

    # world acceleration: a = v_dot * heading + v * w * left  (planar, yaw-only)
    v_dot = np.gradient(v, dt)
    a_world = v_dot[:, None] * heading + (v * w)[:, None] * left

    # body-frame IMU under pypose convention a_world = R @ acc - g
    # => acc_body = R^T (a_world + g); R is yaw rotation about z.
    c, s = np.cos(yaw), np.sin(yaw)
    sf = a_world + GRAVITY  # specific force in world
    acc_body = np.empty_like(sf)
    acc_body[:, 0] = c * sf[:, 0] + s * sf[:, 1]
    acc_body[:, 1] = -s * sf[:, 0] + c * sf[:, 1]
    acc_body[:, 2] = sf[:, 2]
    gyr_body = np.stack([np.zeros_like(w), np.zeros_like(w), w], axis=1)

    # imperfect sensor: constant bias + white noise (what the network corrects)
    acc_meas = acc_body + args.acc_bias + rng.normal(0, args.acc_noise, acc_body.shape)
    gyr_meas = gyr_body + args.gyr_bias + rng.normal(0, args.gyr_noise, gyr_body.shape)

    out = args.out
    os.makedirs(os.path.join(out, "points", "data"), exist_ok=True)

    # imu.csv : ts, ax,ay,az, wx,wy,wz
    imu_csv = np.concatenate([t[:, None], acc_meas, gyr_meas], axis=1)
    np.savetxt(os.path.join(out, "imu.csv"), imu_csv, delimiter=",", fmt="%.9f")

    # ground-truth pose per scan (nearest IMU sample), 7-col x,y,z,qx,qy,qz,qw
    scan_idx = np.searchsorted(t, scan_ts, side="left").clip(0, N - 1)
    gt_pos = pos[scan_idx]
    gt_quat = quat_from_yaw(yaw[scan_idx])
    gt = np.concatenate([gt_pos, gt_quat], axis=1)
    np.savetxt(os.path.join(out, "gt_pose.csv"), gt, delimiter=",", fmt="%.9f")

    # scan timestamps
    np.savetxt(os.path.join(out, "points", "timestamps.txt"), scan_ts, fmt="%.9f")

    # a static world point cloud, re-observed (transformed into the body frame)
    # at each scan pose. Only the ICP/PGO path consumes these; harmless otherwise.
    world = np.concatenate([
        rng.uniform([-20, -20, -2], [20, 20, 4], size=(args.n_points * 3 // 4, 3)),
        np.stack([  # a ring of structure so registration has signal
            15 * np.cos(np.linspace(0, 2 * np.pi, args.n_points // 4)),
            15 * np.sin(np.linspace(0, 2 * np.pi, args.n_points // 4)),
            rng.uniform(-1, 3, args.n_points // 4),
        ], axis=1),
    ], axis=0)
    for k in range(n_scan):
        ci, si = np.cos(-yaw[scan_idx[k]]), np.sin(-yaw[scan_idx[k]])
        Rz_inv = np.array([[ci, -si, 0], [si, ci, 0], [0, 0, 1]])
        local = (world - gt_pos[k]) @ Rz_inv.T
        rec = np.zeros(local.shape[0], dtype=LIDAR_DTYPE)
        rec["x"], rec["y"], rec["z"] = local[:, 0], local[:, 1], local[:, 2]
        rec["intensity"] = 1.0
        rec.tofile(os.path.join(out, "points", "data", f"{k:06d}.bin"))

    print(f"[gen] wrote {n_scan} scans / {N} IMU samples to {out}")
    print(f"[gen] regimes -> lin speed range [{v.min():.2f},{v.max():.2f}] m/s, "
          f"yaw-rate range [{w.min():.2f},{w.max():.2f}] rad/s")


if __name__ == "__main__":
    main()
