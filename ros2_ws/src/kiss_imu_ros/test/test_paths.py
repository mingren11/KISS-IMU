import importlib


def test_ensure_src_on_path_imports_upstream(monkeypatch, tmp_path):
    # KISS_IMU_SRC 指向真实仓库 src，确保上游模块可导入
    import kiss_imu_ros.paths as paths
    paths.ensure_src_on_path()
    # 上游核心模块应能导入
    assert importlib.import_module('models.imu_net') is not None
    assert importlib.import_module('training.integrator') is not None
