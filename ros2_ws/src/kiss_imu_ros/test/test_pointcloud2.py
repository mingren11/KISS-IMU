import numpy as np
from kiss_imu_ros.pointcloud2 import xyz_from_arrays, voxel_downsample
from kiss_imu_ros.pointcloud2 import normalize_per_point_times, voxel_downsample_indexed


def test_xyz_from_structured_filters_nonfinite():
    dt = np.dtype([('x', np.float32), ('y', np.float32), ('z', np.float32), ('intensity', np.float32)])
    a = np.zeros((4,), dtype=dt)
    a['x'] = [1.0, np.nan, 3.0, np.inf]
    a['y'] = [1.0, 2.0, 3.0, 4.0]
    a['z'] = [1.0, 2.0, 3.0, 4.0]
    pts = xyz_from_arrays(a)
    assert pts.shape == (2, 3) and pts.dtype == np.float64
    assert np.allclose(pts[:, 0], [1.0, 3.0])


def test_voxel_downsample_reduces_duplicates():
    pts = np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [5.0, 5.0, 5.0]], dtype=np.float64)
    out = voxel_downsample(pts, voxel=1.0)
    assert out.shape[0] == 2   # first two land in the same voxel


def test_normalize_velodyne_relative_seconds():
    t = np.linspace(0.0, 0.1, 50)            # velodyne: relative seconds
    out = normalize_per_point_times(t)
    assert out.shape == (50,)
    assert out.min() == 0.0 and abs(out.max() - 1.0) < 1e-12
    assert np.all(np.diff(out) >= 0)


def test_normalize_livox_offset_nanoseconds():
    t = np.arange(0, 100000, 2000, dtype=np.float64)  # livox: large ns offsets
    out = normalize_per_point_times(t)
    assert abs(out[0]) < 1e-12 and abs(out[-1] - 1.0) < 1e-12
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_normalize_degenerate_all_equal_returns_zeros():
    t = np.full(10, 5.0)
    out = normalize_per_point_times(t)
    assert out.shape == (10,) and np.all(out == 0.0)


def test_normalize_empty():
    out = normalize_per_point_times(np.zeros((0,)))
    assert out.shape == (0,)


def test_voxel_downsample_indexed_returns_aligned_indices():
    pts = np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [5.0, 5.0, 5.0]], dtype=np.float64)
    ds, idx = voxel_downsample_indexed(pts, voxel=1.0)
    assert ds.shape[0] == 2
    assert idx.shape[0] == 2
    assert np.allclose(ds, pts[idx])


def test_voxel_downsample_indexed_disabled_returns_all():
    pts = np.random.randn(7, 3)
    ds, idx = voxel_downsample_indexed(pts, voxel=0.0)
    assert ds.shape[0] == 7 and idx.shape[0] == 7
