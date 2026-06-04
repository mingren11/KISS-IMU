import importlib


def test_ensure_src_on_path_imports_upstream():
    # KISS_IMU_SRC points at the real repo src so upstream modules import
    import kiss_imu_ros.paths as paths
    paths.ensure_src_on_path()
    # upstream core modules should be importable
    assert importlib.import_module('models.imu_net') is not None
    assert importlib.import_module('training.integrator') is not None
