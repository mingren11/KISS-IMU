"""Livox CustomMsg point conversion (pure, ROS-free, unit-tested).

A Livox `livox_ros_driver2/msg/CustomMsg` carries per-point `offset_time`
(ns within the sweep) which is ideal for deskew. This converts its points to
the (xyz, normalized-time) pair the OnlineEstimator/LOModule consume."""
import numpy as np

from .pointcloud2 import normalize_per_point_times


def custom_points_to_xyz_t(points, drop_zeros: bool = True):
    """Convert Livox CustomMsg points -> (xyz (N,3) float64, t_norm (N,) float64).

    Drops exact (0,0,0) invalid returns (when drop_zeros) and non-finite xyz,
    then normalizes the surviving per-point offset_time to [0,1]."""
    if len(points) == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0,), dtype=np.float64)
    xyz = np.array([(p.x, p.y, p.z) for p in points], dtype=np.float64)
    off = np.array([p.offset_time for p in points], dtype=np.float64)
    mask = np.isfinite(xyz).all(axis=1)
    if drop_zeros:
        mask &= ~(xyz == 0.0).all(axis=1)
    xyz = xyz[mask]
    off = off[mask]
    return xyz, normalize_per_point_times(off)
