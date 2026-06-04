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


def test_build_window_sample_dts_invariant():
    accels, gyros, imu_ts, scan0, scan1 = _fake_window()
    s = build_window_sample(accels, gyros, imu_ts, scan0, scan1)
    dts = s['imu_dts'][0].numpy()
    expected = np.diff(imu_ts)
    # dts[0] duplicates dts[1]; dts[k] = ts[k]-ts[k-1] for k>=1
    assert np.allclose(dts[1:], expected, atol=1e-5)
    assert np.isclose(dts[0], dts[1], atol=1e-7)


def test_raw_corrector_passthrough_gyros_and_dts():
    accels, gyros, imu_ts, scan0, scan1 = _fake_window()
    s = build_window_sample(accels, gyros, imu_ts, scan0, scan1)
    corr = RawCorrector().correct(s)
    assert torch.allclose(corr['gyros_corr'][0].cpu().double(), torch.from_numpy(gyros))
    assert len(corr['dts']) == 1 and corr['dts'][0].shape == (20,)


def test_build_window_sample_threads_scan1_ts():
    accels, gyros, imu_ts, scan0, scan1 = _fake_window()
    t = np.linspace(0.0, 1.0, scan1.shape[0])
    s = build_window_sample(accels, gyros, imu_ts, scan0, scan1, scan1_ts=t)
    assert len(s['scan1_ts']) == 1
    assert s['scan1_ts'][0].shape == (scan1.shape[0],)
    assert torch.allclose(s['scan1_ts'][0].cpu().double(), torch.from_numpy(t))
