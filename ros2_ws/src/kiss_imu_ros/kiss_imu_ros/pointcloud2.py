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
    mask = np.isfinite(xyz).all(axis=1)
    return xyz[mask]
