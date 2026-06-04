import numpy as np
from kiss_imu_ros.ring_buffer import ImuBuffer


def test_pop_window_returns_interval_inclusive_upper():
    buf = ImuBuffer(maxlen=100)
    for i in range(10):
        buf.append(t=float(i), acc=(i, 0.0, 0.0), gyro=(0.0, 0.0, float(i)))
    ts, acc, gyro = buf.pop_window(t0=2.0, t1=5.0)
    # (2,5] -> t = 3,4,5
    assert ts.tolist() == [3.0, 4.0, 5.0]
    assert acc.shape == (3, 3) and gyro.shape == (3, 3)
    assert np.allclose(acc[:, 0], [3.0, 4.0, 5.0])


def test_pop_window_discards_consumed_samples():
    buf = ImuBuffer(maxlen=100)
    for i in range(6):
        buf.append(t=float(i), acc=(0.0, 0.0, 0.0), gyro=(0.0, 0.0, 0.0))
    buf.pop_window(t0=-1.0, t1=2.0)   # consumes t=0,1,2
    ts, _, _ = buf.pop_window(t0=2.0, t1=10.0)
    assert ts.tolist() == [3.0, 4.0, 5.0]


def test_pop_window_empty_when_no_samples_in_range():
    buf = ImuBuffer(maxlen=100)
    buf.append(t=0.0, acc=(0.0, 0.0, 0.0), gyro=(0.0, 0.0, 0.0))
    ts, acc, gyro = buf.pop_window(t0=10.0, t1=20.0)
    assert ts.shape == (0,) and acc.shape == (0, 3) and gyro.shape == (0, 3)
