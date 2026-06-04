"""PointCloud2 <-> numpy helpers (pure-numpy parts are unit-tested without ROS)."""
import numpy as np


def xyz_from_arrays(structured) -> np.ndarray:
    """Take x/y/z from a structured array; return (N,3) float64 with NaN/Inf dropped."""
    xyz = np.stack([structured['x'], structured['y'], structured['z']], axis=-1).astype(np.float64)
    mask = np.isfinite(xyz).all(axis=1)
    return xyz[mask]


def normalize_per_point_times(t) -> np.ndarray:
    """Normalize per-point times to [0,1] within the scan (the kiss_icp deskew
    convention). Works for any units (relative seconds, offset ns, absolute):
    (t - min) / (max - min). Degenerate (all equal) or empty -> zeros/empty."""
    t = np.asarray(t, dtype=np.float64)
    if t.size == 0:
        return t
    lo = float(t.min())
    span = float(t.max()) - lo
    if span <= 0.0:
        return np.zeros_like(t)
    return (t - lo) / span


def voxel_downsample_indexed(pts: np.ndarray, voxel: float):
    """Deterministic voxel downsample; return (downsampled_pts, kept_indices) so a
    caller can downsample an aligned per-point array (e.g. timestamps) identically."""
    if pts.size == 0 or voxel <= 0:
        return pts, np.arange(pts.shape[0])
    grid = np.floor(pts / voxel).astype(np.int64)
    _, idx = np.unique(grid, axis=0, return_index=True)
    idx = np.sort(idx)
    return pts[idx], idx


def voxel_downsample(pts: np.ndarray, voxel: float) -> np.ndarray:
    """Deterministic voxel downsample (unique grid cell, sorted) — mirrors upstream."""
    return voxel_downsample_indexed(pts, voxel)[0]


def xyz_from_pointcloud2(msg, field_names=('x', 'y', 'z'), time_field=None):
    """Parse a sensor_msgs/PointCloud2 into (xyz (N,3) float64, t_norm (N,) or None).

    If ``time_field`` is given AND present in the cloud, per-point times are read,
    finite-aligned with the kept xyz, and normalized to [0,1] (kiss_icp deskew
    convention). Otherwise t_norm is None. Imports ROS lazily."""
    from sensor_msgs_py import point_cloud2
    have_time = bool(time_field) and time_field in [f.name for f in msg.fields]
    read_fields = tuple(field_names) + ((time_field,) if have_time else ())
    structured = point_cloud2.read_points(msg, field_names=read_fields, skip_nans=True)
    xyz = np.stack([structured['x'], structured['y'], structured['z']], axis=-1).astype(np.float64)
    mask = np.isfinite(xyz).all(axis=1)
    xyz = xyz[mask]
    if have_time:
        t = np.asarray(structured[time_field], dtype=np.float64)[mask]
        return xyz, normalize_per_point_times(t)
    return xyz, None
