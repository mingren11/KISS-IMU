import numpy as np
from kiss_imu_ros.pointcloud2 import xyz_from_arrays, voxel_downsample


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
