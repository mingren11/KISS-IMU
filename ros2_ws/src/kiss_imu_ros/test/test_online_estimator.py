import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pypose")
pytest.importorskip("small_gicp")

from kiss_imu_ros.online_estimator import OnlineEstimator, EstimatorConfig


def _synth_scan(n=800, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(-20, 20, size=(n, 3)).astype(np.float64)


def _synth_imu(T=20, t0=0.0, dt=0.005):
    ts = t0 + np.arange(T) * dt
    accels = np.tile([0.0, 0.0, 9.81], (T, 1)).astype(np.float64)  # static: cancels gravity only
    gyros = np.zeros((T, 3), dtype=np.float64)
    return accels, gyros, ts


def test_step_returns_finite_se3_pose():
    cfg = EstimatorConfig(lo_model='small_gicp', device='cpu')
    est = OnlineEstimator(cfg)
    r0 = est.step(_synth_scan(seed=0), *_synth_imu(t0=0.0))   # bootstrap
    assert r0.pose.shape == (7,)
    r1 = est.step(_synth_scan(seed=1), *_synth_imu(t0=0.1))
    assert r1.pose.shape == (7,)
    assert np.all(np.isfinite(r1.pose))
    q = r1.pose[3:]
    assert abs(np.linalg.norm(q) - 1.0) < 1e-3


def test_pose_advances_over_steps():
    cfg = EstimatorConfig(lo_model='small_gicp', device='cpu')
    est = OnlineEstimator(cfg)
    poses = []
    for k in range(4):
        r = est.step(_synth_scan(seed=k), *_synth_imu(t0=0.1 * k))
        poses.append(r.pose[:3])
    assert all(np.all(np.isfinite(p)) for p in poses)


def test_divergent_solve_falls_back_to_finite_pose(monkeypatch):
    import pypose as pp
    import kiss_imu_ros.online_estimator as oe

    def _nan_optimize(*args, **kwargs):
        nan_nodes = pp.SE3(torch.full((2, 7), float('nan')))
        return nan_nodes, torch.zeros((2, 3))

    cfg = EstimatorConfig(lo_model='small_gicp', device='cpu')
    est = OnlineEstimator(cfg)
    est.step(_synth_scan(seed=0), *_synth_imu(t0=0.0))   # bootstrap
    monkeypatch.setattr(oe, 'optimize', _nan_optimize)
    r = est.step(_synth_scan(seed=1), *_synth_imu(t0=0.1))
    assert r.diverged is True
    assert np.all(np.isfinite(r.pose))         # fell back to IMU node, not NaN
    assert np.all(np.isfinite(est.anchor_pose.numpy()))
