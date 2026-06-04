import numpy as np
import torch
from kiss_imu_ros.imu_corrector import build_window_sample, RawCorrector


def _fake_window():
    T = 20
    accels = np.random.randn(T, 3).astype(np.float64)
    gyros = np.random.randn(T, 3).astype(np.float64)
    imu_ts = np.linspace(0.0, 0.1, T)
    scan0 = np.random.randn(500, 3).astype(np.float64)
    scan1 = np.random.randn(500, 3).astype(np.float64)
    return accels, gyros, imu_ts, scan0, scan1


def test_build_window_sample_shapes_and_keys():
    accels, gyros, imu_ts, scan0, scan1 = _fake_window()
    s = build_window_sample(accels, gyros, imu_ts, scan0, scan1)
    assert s['accels'].shape == (1, 20, 3)
    assert s['gyros'].shape == (1, 20, 3)
    assert s['imu_dts'].shape == (1, 20)
    assert s['valid_length'].tolist() == [20]
    assert isinstance(s['scan0'], list) and s['scan0'][0].shape == (500, 3)
    assert len(s['scan1']) == 1 and len(s['scan1_ts']) == 1


def test_raw_corrector_passthrough_structure():
    accels, gyros, imu_ts, scan0, scan1 = _fake_window()
    s = build_window_sample(accels, gyros, imu_ts, scan0, scan1)
    corr = RawCorrector().correct(s)
    assert set(['accels_corr', 'gyros_corr', 'acc_cov', 'gyr_cov', 'valid_length', 'dts']) <= set(corr)
    assert len(corr['accels_corr']) == 1 and corr['accels_corr'][0].shape == (20, 3)
    assert corr['acc_cov'] is None and corr['gyr_cov'] is None
    # raw = no correction: passthrough equals input
    assert torch.allclose(corr['accels_corr'][0].cpu().double(),
                          torch.from_numpy(accels))
