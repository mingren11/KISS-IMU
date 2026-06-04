"""Pluggable IMU correction stage.

RawCorrector = identity passthrough (the raw_pvgo baseline). A future
LearnedCorrector will wrap IMUNet and return the same dict structure
(see src/models/imu_net.py forward()), so OnlineEstimator stays unchanged.
"""
import numpy as np
import torch


def build_window_sample(accels, gyros, imu_ts, scan0, scan1, scan1_ts=None):
    """Build a batch-of-1 sample dict with exactly the keys IMUNet.forward and
    LOModule.forward read. dts[k] = ts[k]-ts[k-1], with dts[0] duplicated from dts[1]."""
    accels = np.asarray(accels, dtype=np.float64)
    gyros = np.asarray(gyros, dtype=np.float64)
    imu_ts = np.asarray(imu_ts, dtype=np.float64)
    T = accels.shape[0]
    if T >= 2:
        dts = np.empty(T, dtype=np.float64)
        dts[1:] = np.diff(imu_ts)
        dts[0] = dts[1]
    else:
        dts = np.full(T, 1e-3, dtype=np.float64)

    a = torch.from_numpy(accels).float().unsqueeze(0)        # (1,T,3)
    g = torch.from_numpy(gyros).float().unsqueeze(0)         # (1,T,3)
    ts = torch.from_numpy(imu_ts).float().unsqueeze(0)       # (1,T)
    dt = torch.from_numpy(dts).float().unsqueeze(0)          # (1,T)
    s0 = torch.from_numpy(np.asarray(scan0, dtype=np.float64)).float()
    s1 = torch.from_numpy(np.asarray(scan1, dtype=np.float64)).float()
    if scan1_ts is None:
        scan1_ts = np.zeros(s1.shape[0], dtype=np.float64)
    s1ts = torch.from_numpy(np.asarray(scan1_ts, dtype=np.float64)).float()

    return {
        'accels': a, 'gyros': g, 'imu_ts': ts, 'imu_dts': dt,
        'valid_length': torch.tensor([T], dtype=torch.long),
        'scan0': [s0], 'scan1': [s1], 'scan1_ts': [s1ts],
    }


class RawCorrector:
    """Identity: emit corrected IMU == raw IMU, no covariance."""

    def correct(self, sample) -> dict:
        vlen = int(sample['valid_length'][0].item())
        return {
            'accels_corr': [sample['accels'][0, :vlen]],
            'gyros_corr': [sample['gyros'][0, :vlen]],
            'acc_cov': None,
            'gyr_cov': None,
            'valid_length': sample['valid_length'],
            'dts': [sample['imu_dts'][0, :vlen]],
        }
