import os
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pypose")
pytest.importorskip("small_gicp")

from kiss_imu_ros.paths import ensure_src_on_path
ensure_src_on_path()
from data.seq_dataset import SeqDataset  # noqa: E402
from kiss_imu_ros.online_estimator import OnlineEstimator, EstimatorConfig  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
SYNTH = os.path.join(REPO, 'data', 'synth')


@pytest.mark.skipif(not os.path.isdir(os.path.join(SYNTH, 'Synth01')),
                    reason="synth dataset not generated; run tools/gen_synth_dataset.py")
def test_replay_synth_produces_finite_trajectory():
    ds = SeqDataset(data_root=SYNTH, data_seq='Synth01', data_type='diter_os')
    cfg = EstimatorConfig(lo_model='small_gicp', device='cpu',
                          gravity=tuple(np.asarray(ds.gravity).reshape(-1).tolist()))
    est = OnlineEstimator(cfg)

    n = min(20, len(ds))
    poses = []
    for i in range(n):
        item = ds[i]
        accels = item['accels'].numpy()
        gyros = item['gyros'].numpy()
        imu_ts = item['imu_ts'].numpy()
        scan1 = item['scan1'].numpy()
        r = est.step(scan1, accels, gyros, imu_ts)
        poses.append(r.pose)
    poses = np.asarray(poses)
    assert poses.shape == (n, 7)
    assert np.all(np.isfinite(poses))
    # estimator must actually move, not return the bootstrap anchor forever
    assert not np.allclose(poses[0], poses[-1])
