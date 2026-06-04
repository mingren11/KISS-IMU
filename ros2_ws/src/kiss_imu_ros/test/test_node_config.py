import pytest
pytest.importorskip("torch")
pytest.importorskip("pypose")
pytest.importorskip("rclpy")


def test_params_to_cfg_maps_extrinsics():
    from kiss_imu_ros.lio_node import params_to_cfg
    params = dict(lo_model='small_gicp', device='cpu', use_submap=False,
                  lm_weight=[1.0, 0.1, 1.0, 0.1, 0.1], gravity=[0.0, 0.0, 9.81],
                  R_I_L=[1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0], T_I_L=[0.1, 0.2, 0.3])
    cfg = params_to_cfg(params)
    assert cfg.lo_model == 'small_gicp'
    assert cfg.R_I_L == [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]
    assert cfg.T_I_L == [0.1, 0.2, 0.3]
