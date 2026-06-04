#!/usr/bin/env bash
# Entry point for the KISS-IMU synthetic CPU demo (run inside the container
# built from docker/Dockerfile.demo).
#
# 1. Generate a tiny synthetic DiTer-OS-style sequence (no external dataset).
# 2. Train the GT-supervision path on CPU (skips LiDAR ICP/PGO).
# Results land under results/ which is bind-mounted back to the host.
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root

# Demo knobs — override via `docker compose run -e EPOCH=20 ...` etc.
EPOCH=${EPOCH:-10}
BATCH_SIZE=${BATCH_SIZE:-4}

echo "[run_demo] generating synthetic dataset -> data/synth/Synth01"
python tools/gen_synth_dataset.py --out data/synth/Synth01

echo "[run_demo] training (CPU, GT supervision, EPOCH=${EPOCH}, BATCH_SIZE=${BATCH_SIZE})"
CUDA_VISIBLE_DEVICES="" \
DATA_DIR="$PWD/data/synth" DATA_TYPE=diter_os \
TRAIN_SEQS=Synth01 VALID_SEQS=Synth01 \
LO_MODEL=small_gicp DEVICE=cpu EPOCH="${EPOCH}" BATCH_SIZE="${BATCH_SIZE}" \
USE_GT=true TRAIN_RATIO=1.0 \
bash scripts/train.sh

echo "[run_demo] done. Trained artifacts:"
find results -name best_model.ckpt -printf '  %h\n' 2>/dev/null || true
