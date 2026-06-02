"""Batch collation for :class:`data.seq_dataset.SeqDataset`.

A "sample" is one scan-to-scan window. IMU streams have variable length per
window, so they are zero-padded to the batch max and the true length is kept in
``valid_length`` (the network/integrator only ever read ``[:valid_length]``).
Point clouds have variable cardinality and are kept as plain Python lists; the
LO module consumes them one element at a time.
"""

import torch


def _pad_stack(seqs, feat_dims):
    """Zero-pad a list of (T_i, *feat_dims) tensors to (B, T_max, *feat_dims)."""
    B = len(seqs)
    t_max = max(int(s.shape[0]) for s in seqs)
    out = torch.zeros((B, t_max, *feat_dims), dtype=seqs[0].dtype)
    for b, s in enumerate(seqs):
        out[b, : s.shape[0]] = s
    return out


def collate_fn(batch):
    accels = _pad_stack([b["accels"] for b in batch], (3,))
    gyros = _pad_stack([b["gyros"] for b in batch], (3,))
    imu_ts = _pad_stack([b["imu_ts"].reshape(-1, 1) for b in batch], (1,)).squeeze(-1)
    imu_dts = _pad_stack([b["imu_dts"].reshape(-1, 1) for b in batch], (1,)).squeeze(-1)
    valid_length = torch.tensor([int(b["accels"].shape[0]) for b in batch], dtype=torch.long)

    out = {
        "accels": accels,                 # (B, Tmax, 3)
        "gyros": gyros,                   # (B, Tmax, 3)
        "imu_ts": imu_ts,                 # (B, Tmax)
        "imu_dts": imu_dts,               # (B, Tmax)
        "valid_length": valid_length,     # (B,)
        # variable-cardinality point clouds: keep as lists (consumed per-window)
        "scan0": [b["scan0"] for b in batch],
        "scan1": [b["scan1"] for b in batch],
        "scan1_ts": [b["scan1_ts"] for b in batch],
        # SE3 ground-truth poses [x,y,z, qx,qy,qz,qw] and start-of-window velocity
        "gt_pose0": torch.stack([b["gt_pose0"] for b in batch], dim=0),   # (B, 7)
        "gt_pose1": torch.stack([b["gt_pose1"] for b in batch], dim=0),   # (B, 7)
        "gt_velocity": torch.stack([b["gt_velocity"] for b in batch], 0), # (B, 3)
    }
    return out
