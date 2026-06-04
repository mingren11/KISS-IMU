import numpy as np
from kiss_imu_ros.livox import custom_points_to_xyz_t


class _P:
    __slots__ = ('x', 'y', 'z', 'offset_time')

    def __init__(self, x, y, z, offset_time):
        self.x, self.y, self.z, self.offset_time = x, y, z, offset_time


def test_converts_and_normalizes_offset_time():
    pts = [_P(1.0, 0.0, 0.0, 0), _P(2.0, 0.0, 0.0, 50_000), _P(3.0, 0.0, 0.0, 100_000)]
    xyz, t = custom_points_to_xyz_t(pts)
    assert xyz.shape == (3, 3) and xyz.dtype == np.float64
    assert np.allclose(xyz[:, 0], [1.0, 2.0, 3.0])
    assert abs(t[0]) < 1e-12 and abs(t[-1] - 1.0) < 1e-12


def test_drops_zero_returns_and_nonfinite():
    pts = [_P(0.0, 0.0, 0.0, 0),
           _P(1.0, 2.0, 3.0, 10_000),
           _P(np.nan, 1.0, 1.0, 20_000),
           _P(4.0, 5.0, 6.0, 30_000)]
    xyz, t = custom_points_to_xyz_t(pts)
    assert xyz.shape == (2, 3)
    assert t.shape == (2,)
    assert np.allclose(xyz[0], [1.0, 2.0, 3.0]) and np.allclose(xyz[1], [4.0, 5.0, 6.0])


def test_keep_zeros_when_disabled():
    pts = [_P(0.0, 0.0, 0.0, 0), _P(1.0, 1.0, 1.0, 10_000)]
    xyz, t = custom_points_to_xyz_t(pts, drop_zeros=False)
    assert xyz.shape == (2, 3)


def test_empty_input():
    xyz, t = custom_points_to_xyz_t([])
    assert xyz.shape == (0, 3) and t.shape == (0,)
