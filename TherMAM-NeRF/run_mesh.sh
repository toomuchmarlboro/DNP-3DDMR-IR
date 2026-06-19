#!/bin/bash
# ---------------------------------------------------------------------------
# Re-run the cohort at a chosen MESH SIZE (default 1.0 mm) to VERIFY that the
# recovered geometry (z_hat, r_hat) is mesh-independent.
#
# NOTE: mesh size is NOT a boundary condition. It is the discretization
# resolution. The convergence study (§6.5) and the 1 mm spot-check already show
# r_hat is unchanged at 1 mm (Patient_10 40->40, Patient_125 5.1->5.0). This
# script just lets you confirm it yourself on a subset.
#
# WARNING: a 1 mm solve is ~25x slower than 3 mm (each forward solve minutes,
# ~70 per patient) -> ~3-4 h/patient. Keep SUBSET small.
#
# The underlying .py is:  mukhmetov_recover.py --mesh-size <mm>
#
# Usage (two tmux panes):
#   pane 0:  ./run_mesh.sh 0 0
#   pane 1:  ./run_mesh.sh 1 1
#
# Optional:  SUBSET=10 MESH=1.0 ./run_mesh.sh 0 0   (defaults shown)
# ---------------------------------------------------------------------------
set -u
GPU="${1:-0}"
OFFSET="${2:-0}"
SUBSET="${SUBSET:-10}"          # small on purpose: 1 mm is expensive
MESH="${MESH:-1.0}"            # element size in mm

PYTHON=/mnt/Data1/Peoples/faiz836b/miniconda3/envs/bioheat/bin/python
cd /mnt/Data1/Peoples/faiz836b/DNP-3DDMR-IR/TherMAM-NeRF || exit 1

OUT="results/cohort_mesh${MESH}mm_gpu${GPU}.csv"
LOG="mesh${MESH}mm_gpu${GPU}.log"

echo "GPU=${GPU}  offset=${OFFSET}  subset=${SUBSET}  mesh=${MESH}mm"
echo "output -> ${OUT}   log -> ${LOG}"
echo "(BCs are PRODUCTION h=10/T_air=20 — only the mesh changes)"

CUDA_VISIBLE_DEVICES="${GPU}" $PYTHON -u mukhmetov_recover.py --real \
    --subset "${SUBSET}" --stride 2 --offset "${OFFSET}" \
    --mesh-size "${MESH}" \
    --out "${OUT}" --resume 2>&1 | tee "${LOG}"
