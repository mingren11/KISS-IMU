"""Sequence dataset for KISS-IMU.

Reconstructed to match the contract the rest of the codebase consumes
(``train.py``, ``models/imu_net.py``, ``models/lo_module.py``, ``models/gmm.py``,
``training/integrator.py`` and ``examples/dataset_layout.md``).

A sequence directory looks like::

    <data_root>/<data_seq>/
      ├── imu.csv            # col0 = timestamp[s], then accel/gyro by index
      ├── gt_pose.csv        # one pose per LiDAR scan (format per --data-type)
      └── points/
          ├── data/000000.bin ...   # packed structured array (lidar_dtype)
          └── timestamps.txt        # one scan timestamp[s] per line

One dataset item is one *scan-to-scan window*: the pair of consecutive scans
``(scan i, scan i+1)`` together with the IMU samples whose timestamp falls in
``[ts_i, ts_{i+1})`` and the ground-truth poses at both ends. ``len(dataset)``
is therefore ``num_scans - 1``.
"""

import os
import glob

import numpy as np
import pandas as pd
import torch
from scipy.spatial.transform import Rotation


# --------------------------------------------------------------------------- #
# Per-dataset configuration. acc_idx / gyr_idx are *column indices* into
# imu.csv (column 0 is always the timestamp in seconds). gt_format selects how
# gt_pose.csv rows are turned into [x,y,z, qx,qy,qz,qw].
# --------------------------------------------------------------------------- #
_DITER_DTYPE = np.dtype([("x", "<f8"), ("y", "<f8"), ("z", "<f8"), ("intensity", "<f8")])
_KITTI_DTYPE = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("intensity", "<f4")])

DATASET_CFG = {
    "diter_os": dict(acc_idx=(1, 2, 3), gyr_idx=(4, 5, 6),
                     gt_format="xyzquat", lidar_dtype=_DITER_DTYPE,
                     R_I_L=np.eye(3), T_I_L=np.zeros(3), gravity=9.81),
    "diter++":  dict(acc_idx=(1, 2, 3), gyr_idx=(4, 5, 6),
                     gt_format="xyzquat", lidar_dtype=_DITER_DTYPE,
                     R_I_L=np.eye(3), T_I_L=np.zeros(3), gravity=9.81),
    "kitti":    dict(acc_idx=(1, 2, 3), gyr_idx=(4, 5, 6),
                     gt_format="kitti12", lidar_dtype=_KITTI_DTYPE,
                     R_I_L=np.eye(3), T_I_L=np.zeros(3), gravity=9.81),
}


def _cfg_for(data_type):
    if data_type not in DATASET_CFG:
        # Fall back to the diter_os layout for unknown types.
        return DATASET_CFG["diter_os"]
    return DATASET_CFG[data_type]


def _parse_gt_row(rows, gt_format):
    """Return (positions (M,3), quats_xyzw (M,4)) from raw gt_pose.csv rows."""
    rows = np.asarray(rows, dtype=np.float64)
    if gt_format == "xyzquat":
        pos = rows[:, 0:3]
        quat = rows[:, 3:7]
        quat = quat / (np.linalg.norm(quat, axis=1, keepdims=True) + 1e-12)
        return pos, quat
    if gt_format == "kitti12":
        # 12 values per row = row-major [R|t] of a 3x4 matrix.
        M = rows.shape[0]
        T = rows[:, :12].reshape(M, 3, 4)
        pos = T[:, :, 3]
        quat = Rotation.from_matrix(T[:, :, :3]).as_quat()  # xyzw
        return pos, quat
    raise ValueError(f"Unknown gt_format: {gt_format}")


class SeqDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, data_seq, data_type, window_size=None):
        self.data_root = data_root
        self.data_seq = data_seq
        self.data_type = data_type
        self.window_size = window_size  # accepted for API parity (eval scripts)
        self.cfg = _cfg_for(data_type)

        seq_dir = os.path.join(data_root, data_seq)

        # ---- IMU stream ----------------------------------------------------
        imu = pd.read_csv(os.path.join(seq_dir, "imu.csv"), header=None).to_numpy(np.float64)
        ai, gi = self.cfg["acc_idx"], self.cfg["gyr_idx"]
        imu_ts = imu[:, 0].astype(np.float64)
        accels = imu[:, list(ai)].astype(np.float64)
        gyros = imu[:, list(gi)].astype(np.float64)

        # Full flat streams: GmmModule fits its motion-regime GMM on these.
        self.imu_ts = imu_ts
        self.accels = accels
        self.gyros = gyros

        # ---- scan timestamps + bin paths ----------------------------------
        pts_dir = os.path.join(seq_dir, "points")
        self.bin_paths = sorted(glob.glob(os.path.join(pts_dir, "data", "*.bin")))
        scan_ts = np.loadtxt(os.path.join(pts_dir, "timestamps.txt"), dtype=np.float64)
        scan_ts = np.atleast_1d(scan_ts)
        n_scan = min(len(self.bin_paths), len(scan_ts))
        self.bin_paths = self.bin_paths[:n_scan]
        self.scan_ts = scan_ts[:n_scan]

        # ---- ground-truth poses (one per scan) ----------------------------
        gt_raw = pd.read_csv(os.path.join(seq_dir, "gt_pose.csv"), header=None).to_numpy(np.float64)
        pos, quat = _parse_gt_row(gt_raw, self.cfg["gt_format"])
        n = min(n_scan, len(pos))
        self.bin_paths = self.bin_paths[:n]
        self.scan_ts = self.scan_ts[:n]
        self.gt_pos = pos[:n]
        self.gt_quat = quat[:n]
        self.gt_pose7 = np.concatenate([self.gt_pos, self.gt_quat], axis=1)  # (n, 7)

        # Per-scan velocity by forward difference of GT position.
        dt_scan = np.diff(self.scan_ts)
        dt_scan = np.clip(dt_scan, 1e-6, None)
        vel = np.zeros_like(self.gt_pos)
        vel[:-1] = np.diff(self.gt_pos, axis=0) / dt_scan[:, None]
        vel[-1] = vel[-2] if n > 1 else 0.0
        self.gt_vel = vel

        # ---- window index ranges (precomputed) -----------------------------
        self._win_imu_idx = []
        for i in range(n - 1):
            t0, t1 = self.scan_ts[i], self.scan_ts[i + 1]
            lo = int(np.searchsorted(imu_ts, t0, side="left"))
            hi = int(np.searchsorted(imu_ts, t1, side="left"))
            self._win_imu_idx.append((lo, hi, t0))
        self.num_windows = len(self._win_imu_idx)

        # ---- extrinsics / gravity / init state ------------------------------
        self.R_I_L = np.asarray(self.cfg["R_I_L"], dtype=np.float64)
        self.T_I_L = np.asarray(self.cfg["T_I_L"], dtype=np.float64)
        self.T_I_G = self.T_I_L  # lo_module uses the name T_I_L for this vector
        self.gravity = torch.tensor([0.0, 0.0, float(self.cfg["gravity"])], dtype=torch.float32)

        self.init = {
            "pos": torch.tensor(self.gt_pos[0], dtype=torch.float32),
            "rot": torch.tensor(self.gt_quat[0], dtype=torch.float32),
            "vel": torch.tensor(self.gt_vel[0], dtype=torch.float32),
        }

    # ----------------------------------------------------------------------- #
    def __len__(self):
        return self.num_windows

    def _load_scan(self, idx):
        scan = np.fromfile(self.bin_paths[idx], dtype=self.cfg["lidar_dtype"])
        pts = np.stack([scan["x"], scan["y"], scan["z"]], axis=-1).astype(np.float64)
        return torch.from_numpy(pts)

    def __getitem__(self, i):
        lo, hi, t0 = self._win_imu_idx[i]
        ts = self.imu_ts[lo:hi]
        acc = self.accels[lo:hi]
        gyr = self.gyros[lo:hi]

        # per-sample dt; first dt measured from the window start timestamp
        dts = np.empty_like(ts)
        if len(ts) > 0:
            dts[1:] = np.diff(ts)
            dts[0] = max(ts[0] - t0, 1e-4)
        dts = np.clip(dts, 1e-4, None)

        scan0 = self._load_scan(i)
        scan1 = self._load_scan(i + 1)
        # per-point timestamps in [0,1] (used only by the kiss_icp deskew path)
        scan1_ts = torch.linspace(0.0, 1.0, scan1.shape[0], dtype=torch.float64)

        return {
            "accels": torch.tensor(acc, dtype=torch.float32),
            "gyros": torch.tensor(gyr, dtype=torch.float32),
            "imu_ts": torch.tensor(ts, dtype=torch.float64),
            "imu_dts": torch.tensor(dts, dtype=torch.float32),
            "scan0": scan0,
            "scan1": scan1,
            "scan1_ts": scan1_ts,
            "gt_pose0": torch.tensor(self.gt_pose7[i], dtype=torch.float32),
            "gt_pose1": torch.tensor(self.gt_pose7[i + 1], dtype=torch.float32),
            "gt_velocity": torch.tensor(self.gt_vel[i], dtype=torch.float32),
        }
