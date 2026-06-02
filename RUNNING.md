# Running KISS-IMU — reproduction & quick-start guide

This document explains how to get KISS-IMU running end-to-end, including a
self-contained **synthetic-data demo** that needs no external dataset and no
GPU. It complements [`README.md`](README.md) (which assumes you already have
the real DiTer-OS / KITTI / MulRan datasets and the full LiDAR stack).

> TL;DR — from the repo root:
> ```bash
> conda activate air-io                       # an env with torch + pypose
> pip install pandas scikit-learn tensorboard joblib
> python tools/gen_synth_dataset.py --out data/synth/Synth01
> CUDA_VISIBLE_DEVICES="" DATA_DIR=$PWD/data/synth DATA_TYPE=diter_os \
>   TRAIN_SEQS=Synth01 VALID_SEQS=Synth01 LO_MODEL=small_gicp \
>   DEVICE=cpu EPOCH=10 BATCH_SIZE=4 USE_GT=true bash scripts/train.sh
> ```

---

## 0. What was fixed to make a fresh clone runnable

A fresh clone of the original repository could not run, for two reasons that
are now resolved in this fork:

1. **The `src/data/` package was missing.** Every entry point imports
   `data.seq_dataset.SeqDataset` / `data.collate.collate_fn` /
   `data.eval_dataset`, but the package was never committed — the `.gitignore`
   rule `data/` (intended for *dataset* directories) also matched the *source
   package* `src/data/`. The `.gitignore` rule is now anchored to the repo root
   (`/data/`), and `src/data/` has been **reconstructed** from the exact
   interface the rest of the code consumes (see
   [§3](#3-the-srcdata-package-data-contract)).

2. **A few hard environment assumptions.** Minimal, behavior-preserving edits:
   - `training/integrator.py` — works with both pypose builds where `gravity`
     is a 3-vector *and* `pypose>=0.6.8` where it is a scalar; also propagates
     the configured `device` (it previously hardcoded `cuda:0`), so CPU runs
     work.
   - `models/lo_module.py`, `utils/overlap_score.py`, `utils/point_module.py` —
     the optional LiDAR backends (`cv2`, `open3d`, `pygicp`, `kiss_icp`,
     `small_gicp`) are now imported behind `try/except`, so the package imports
     even when those (hard-to-build) C++ libraries are absent. They are only
     required when that backend is actually selected.

---

## 1. Environment

KISS-IMU needs PyTorch + pypose at minimum. The Docker image in
[`README.md`](README.md#-docker-recommended) ships everything; for a native
setup the core requirements are:

```bash
# core (always needed)
pip install numpy pandas scipy scikit-learn matplotlib tqdm pyyaml joblib
pip install torch torchvision pypose tensorboard

# LiDAR odometry backends (ONLY needed for the full ICP/PGO pseudo-label path)
pip install open3d opencv-python kiss-icp small_gicp pygicp
```

See [`requirements.txt`](requirements.txt) for the full list.

> **GPU note.** Training uses `--device cuda:0` by default. If you have no
> usable GPU (or very little free VRAM), run on CPU by passing
> `DEVICE=cpu` **and** exporting `CUDA_VISIBLE_DEVICES=""` so PyTorch does not
> try to initialize CUDA.

---

## 2. Synthetic-data demo (no dataset, no GPU)

`tools/gen_synth_dataset.py` writes a tiny DiTer-OS-style sequence so you can
exercise the whole training pipeline without any real data.

```bash
python tools/gen_synth_dataset.py --out data/synth/Synth01
```

It produces a physically-consistent, planar (yaw-only) trajectory:

- IMU is derived from the trajectory under pypose's convention
  (`a_world = R·acc − g`, i.e. `acc_body = Rᵀ(a_world + g)`), so integrating the
  raw IMU already approximates the ground truth and the network only has to
  learn small corrections.
- The path is built from several **distinct motion regimes** (slow/fast
  straights, gentle/sharp turns), so the GMM motion-regime clustering finds
  more than one component (BIC typically picks K ≈ 6–7).

Output layout (matches [`examples/dataset_layout.md`](examples/dataset_layout.md)):

```
data/synth/Synth01/
├── imu.csv            # ts, ax,ay,az, wx,wy,wz   (200 Hz)
├── gt_pose.csv        # x,y,z,qx,qy,qz,qw        (one row per scan)
└── points/
    ├── data/000000.bin …   # packed float64 (x,y,z,intensity)
    └── timestamps.txt      # 10 Hz scan timestamps
```

Useful flags: `--duration`, `--acc-bias`, `--gyr-bias`, `--acc-noise`,
`--gyr-noise`, `--n-points`, `--seed`.

### Run training (GT-supervision path, CPU)

```bash
CUDA_VISIBLE_DEVICES="" \
DATA_DIR=$PWD/data/synth DATA_TYPE=diter_os \
TRAIN_SEQS=Synth01 VALID_SEQS=Synth01 \
LO_MODEL=small_gicp DEVICE=cpu EPOCH=10 BATCH_SIZE=4 \
USE_GT=true TRAIN_RATIO=1.0 \
bash scripts/train.sh
```

| Variable | Why this value for the demo |
| --- | --- |
| `CUDA_VISIBLE_DEVICES=""` + `DEVICE=cpu` | Run on CPU (no GPU required). |
| `USE_GT=true` | Supervise with GT poses and **skip LiDAR ICP/PGO** — so the demo needs none of the C++ LiDAR libs. This is the documented GT-supervision ablation; it exercises the IMU network, the GMM motion-balanced sampler, the frequency gate, the covariance-weighted losses, and IMU preintegration. |
| `LO_MODEL=small_gicp` | Not actually invoked under `USE_GT=true`; chosen so the `LOModule` constructor stays trivial. |
| `EPOCH=10` | A few epochs are enough to see loss/metrics move; bump it up for a nicer trajectory. |

### Look at the results

Everything lands under `results/…/<train>_valid_<valid>/<ratio>/`. Locate it:

```bash
RD=$(find results -name best_model.ckpt -printf '%h\n' | head -1); echo "$RD"
```

- `best_model.ckpt` — trained network (+ per-epoch ckpts under `train/`,`valid/`).
- `train/Synth01/ckpt/000N.png`, `valid/Synth01/ckpt/000N.png` — top-down
  trajectory plots (GT vs ICP/PGO/label).
- `gmm.joblib` + `gmm_initial_distribution.png` + `gmm_initial_stats.txt` —
  the fitted motion-regime GMM and its component weights/means.
- `monitoring/` — TensorBoard logs: `tensorboard --logdir "$RD/monitoring"`.

The console also prints per-epoch validation metrics:

```
[Summary] RTE in Synth01 at epoch N: X.XXX m
[Summary] RRE in Synth01 at epoch N: XX.XXX deg
```

> These synthetic numbers only demonstrate that the pipeline runs — they are
> **not** comparable to the paper's results on real data.

---

## 3. The `src/data/` package (data contract)

The reconstructed loader matches what the rest of the codebase expects. Useful
if you want to plug in your own dataset.

`SeqDataset(data_root, data_seq, data_type, window_size=None)` treats one
*scan-to-scan window* as one item — the pair `(scan i, scan i+1)` plus the IMU
samples in `[ts_i, ts_{i+1})` and the GT poses at both ends; `len(dataset) =
num_scans − 1`.

**Per-item dict** (then batched by `collate_fn`):

| key | shape | notes |
| --- | --- | --- |
| `accels`, `gyros` | `(T, 3)` | IMU in the window (zero-padded to `(B, Tmax, 3)` in the batch) |
| `imu_ts`, `imu_dts` | `(T,)` | timestamps & per-sample dt |
| `scan0`, `scan1` | `(N, 3)` | consecutive LiDAR scans (kept as lists in the batch) |
| `scan1_ts` | `(N,)` | per-point timestamps (kiss_icp deskew only) |
| `gt_pose0`, `gt_pose1` | `(7,)` | `[x,y,z, qx,qy,qz,qw]` |
| `gt_velocity` | `(3,)` | start-of-window velocity |

**Dataset attributes:** `.init` `{pos, rot(xyzw), vel}`, `.gravity`,
`.T_I_G` / `.R_I_L` (IMU↔LiDAR extrinsics), `.data_root/.data_seq/.data_type`,
and the flat IMU streams `.imu_ts/.accels/.gyros` (the GMM fits on these).

Per-dataset column indices, the GT format, the LiDAR dtype, the extrinsics and
gravity live in `DATASET_CFG` at the top of
[`src/data/seq_dataset.py`](src/data/seq_dataset.py). Add an entry there for a
new `--data-type`.

---

## 4. Beyond the demo

- **Full pseudo-label path** (drop `USE_GT`): runs real LiDAR ICP + PGO. This
  needs working `open3d` / `kiss-icp` / `pygicp` / `small_gicp` and real point
  clouds, and benefits greatly from a GPU. See
  [`README.md`](README.md#-training) for the dataset layout and all
  hyper-parameters.
- **Evaluation / inference / t-SNE:** `scripts/evaluate.sh`,
  `scripts/inference.sh`, `scripts/tsne_encoder.sh` — point `CKPT` at the
  `best_model.ckpt` produced above. See the corresponding sections in
  [`README.md`](README.md).
