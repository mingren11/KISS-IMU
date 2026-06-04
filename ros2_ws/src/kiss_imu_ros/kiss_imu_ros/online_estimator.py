"""Streaming LiDAR-inertial odometry estimator (ROS-free).

Refactored from src/inference.py's per-window loop. One step() == one scan +
the IMU interval that precedes it. Phase 1 fuses with a 2-node PVGO and a raw
(identity) IMU corrector; the corrector is pluggable for a learned model later.
"""
from dataclasses import dataclass, field

import numpy as np
import torch

from .paths import ensure_src_on_path
ensure_src_on_path()

import pypose as pp                                  # noqa: E402
from training.integrator import IMUIntegrator        # noqa: E402
from models.lo_module import LOModule                # noqa: E402
from models.pvgo import optimize                     # noqa: E402

from .imu_corrector import build_window_sample, RawCorrector  # noqa: E402


@dataclass
class EstimatorConfig:
    lo_model: str = 'kiss_icp'
    device: str = 'cuda:0'
    use_submap: bool = False
    lm_weight: tuple = (1.0, 0.1, 1.0, 0.1, 0.1)
    gravity: tuple = (0.0, 0.0, 9.81)
    R_I_L: list = field(default_factory=lambda: [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    T_I_L: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    init_pose: tuple = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)  # xyz + quat xyzw
    init_vel: tuple = (0.0, 0.0, 0.0)


@dataclass
class OdomResult:
    pose: np.ndarray       # (7,) xyz + quat xyzw
    vel: np.ndarray        # (3,)
    overlap: float
    diverged: bool = False


class OnlineEstimator:
    def __init__(self, cfg: EstimatorConfig):
        self.cfg = cfg
        self.device = cfg.device
        self.gravity = torch.tensor(list(cfg.gravity), dtype=torch.float32)
        self.lm_weight = list(cfg.lm_weight)
        self.corrector = RawCorrector()

        init_pose7 = torch.tensor(list(cfg.init_pose), dtype=torch.float32)
        init_state = {
            'pos': torch.tensor(list(cfg.init_pose[:3]), dtype=torch.float32),
            'rot': torch.tensor(list(cfg.init_pose[3:]), dtype=torch.float32),
            'vel': torch.tensor(list(cfg.init_vel), dtype=torch.float32),
        }
        self.integrator = IMUIntegrator(init_state=init_state,
                                        gravity=self.gravity, device=self.device)
        self.lo_model = LOModule(
            lo_model=cfg.lo_model,
            T_I_L=np.asarray(cfg.T_I_L, dtype=np.float64),
            R_I_L=np.asarray(cfg.R_I_L, dtype=np.float64),
            init_state=init_pose7,
            device_id=self.device,
            use_submap=cfg.use_submap,
        )

        self.anchor_pose = init_pose7.clone()
        self.anchor_vel = torch.tensor(list(cfg.init_vel), dtype=torch.float32)
        self._prev_scan = None

    @torch.no_grad()
    def step(self, scan_xyz, accels, gyros, imu_ts, scan1_ts=None) -> OdomResult:
        scan_xyz = np.asarray(scan_xyz, dtype=np.float64)

        if self._prev_scan is None or len(accels) < 2:
            self._prev_scan = scan_xyz
            return OdomResult(pose=self.anchor_pose.numpy().copy(),
                              vel=self.anchor_vel.numpy().copy(), overlap=0.0)

        sample = build_window_sample(accels, gyros, imu_ts, self._prev_scan, scan_xyz,
                                     scan1_ts=scan1_ts)
        corr = self.corrector.correct(sample)
        # move corrected IMU to the estimator device (IMUNet.forward did this; RawCorrector does not)
        dev = self.device
        corr['accels_corr'] = [a.to(dev) for a in corr['accels_corr']]
        corr['gyros_corr'] = [g.to(dev) for g in corr['gyros_corr']]
        corr['dts'] = [d.to(dev) for d in corr['dts']]
        if corr['acc_cov'] is not None:
            corr['acc_cov'] = [c.to(dev) for c in corr['acc_cov']]
        if corr['gyr_cov'] is not None:
            corr['gyr_cov'] = [c.to(dev) for c in corr['gyr_cov']]

        init_pos = self.anchor_pose[:3].to(self.device).float()
        init_rot = self.anchor_pose[3:].to(self.device).float()
        init_rot = init_rot / (torch.linalg.norm(init_rot) + 1e-12)
        init_v = self.anchor_vel.to(self.device).float()
        # Fresh dict per integrate() call: the integrator mutates last_state in place.
        init_state = {'rot': pp.SO3(init_rot), 'vel': init_v, 'pos': init_pos, 'cov': None}

        imu_states = self.integrator.integrate(
            init=init_state, dts=corr['dts'],
            accels=corr['accels_corr'], gyros=corr['gyros_corr'],
            cov_accels=corr['acc_cov'], cov_gyros=corr['gyr_cov'],
            motion_mode=False)
        zero_init = {'rot': pp.SO3(init_rot),
                     'vel': torch.zeros((1, 3), device=self.device),
                     'pos': torch.zeros((1, 3), device=self.device), 'cov': None}
        imu_d = self.integrator.integrate(
            init=zero_init, dts=corr['dts'],
            accels=corr['accels_corr'], gyros=corr['gyros_corr'],
            cov_accels=corr['acc_cov'], cov_gyros=corr['gyr_cov'],
            motion_mode=True)

        imu_nodes = pp.SE3(torch.cat([imu_states['pos'], imu_states['rot'].tensor()], dim=-1)).to(self.device)
        imu_vels = imu_states['vel'].to(self.device)
        imu_dts = torch.stack([d.sum() for d in corr['dts']]).unsqueeze(-1).to(self.device)

        icp_poses, icp_motions, icp_overlap = self.lo_model(
            sample, pp.SE3(self.anchor_pose.to(self.device)))

        pgo_poses, pgo_vels = optimize(
            nodes=imu_nodes, vels=imu_vels,
            icp_factors=icp_motions,
            imu_drots=imu_d['rot'], imu_dvels=imu_d['vel'],
            imu_dtrans=imu_d['pos'], imu_dts=imu_dts,
            weights=self.lm_weight, gravity=self.gravity,
            icp_weights=None, imu_weights=None, device=self.device)

        new_pose = pgo_poses.tensor()[-1].detach().cpu()
        new_vel = pgo_vels[-1].detach().cpu()
        diverged = not (torch.isfinite(new_pose).all() and torch.isfinite(new_vel).all())
        if diverged:
            # divergent solve -> keep going on the IMU-integrated node instead of
            # poisoning every future step through the carried anchor
            new_pose = imu_nodes.tensor()[-1].detach().cpu()
            new_vel = imu_vels[-1].detach().cpu()

        self.anchor_pose = new_pose
        self.anchor_vel = new_vel
        self._prev_scan = scan_xyz

        return OdomResult(
            pose=self.anchor_pose.numpy().copy(),
            vel=self.anchor_vel.numpy().copy(),
            overlap=float(np.asarray(icp_overlap).reshape(-1)[-1]),
            diverged=diverged,
        )
