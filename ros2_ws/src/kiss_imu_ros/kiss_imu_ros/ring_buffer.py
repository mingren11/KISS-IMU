"""Thread-safe IMU ring buffer with time-window extraction."""
import threading
from collections import deque

import numpy as np


class ImuBuffer:
    def __init__(self, maxlen: int = 4000):
        self._buf = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def append(self, t: float, acc, gyro) -> None:
        with self._lock:
            self._buf.append((float(t),
                              float(acc[0]), float(acc[1]), float(acc[2]),
                              float(gyro[0]), float(gyro[1]), float(gyro[2])))

    def pop_window(self, t0: float, t1: float):
        """Return samples with t in (t0, t1] sorted ascending; drop samples <= t1."""
        with self._lock:
            rows = [r for r in self._buf if t0 < r[0] <= t1]
            # drop everything up to and including t1
            kept = deque((r for r in self._buf if r[0] > t1), maxlen=self._buf.maxlen)
            self._buf = kept
        if not rows:
            return (np.zeros((0,), dtype=np.float64),
                    np.zeros((0, 3), dtype=np.float64),
                    np.zeros((0, 3), dtype=np.float64))
        rows.sort(key=lambda r: r[0])
        arr = np.asarray(rows, dtype=np.float64)
        return arr[:, 0].copy(), arr[:, 1:4].copy(), arr[:, 4:7].copy()
