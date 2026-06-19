#!/bin/bash
# ---------------------------------------------------------------------------
# Re-run the cohort with CALIBRATED boundary conditions (h=5, T_air=24) so the
# absolute FEM skin temperature is realistic (~29 C, matching real DMR-IR IR),
# instead of the production h=10 / T_air=20 (which gives ~24 C, too cold).
#
# Forward model is otherwise identical (Mukhmetov bioheat: 37 C chest wall +
# convective skin). Bilateral cost unchanged. Outputs go to NEW CSVs so the
# production results are not overwritten — letting you diff before/after.
#
# Usage (two tmux panes):
#   pane 0:  ./run_calibBC.sh 0 0
#   pane 1:  ./run_calibBC.sh 1 1
#
# Optional: SUBSET=122 ./run_calibBC.sh 0 0   (default SUBSET=40 balanced)
# ---------------------------------------------------------------------------
set -u
GPU="${1:-0}"
OFFSET="${2:-0}"
SUBSET="${SUBSET:-40}"
H_CONV="${H_CONV:-5}"
T_AIR="${T_AIR:-24}"

PYTHON=/mnt/Data1/Peoples/faiz836b/miniconda3/envs/bioheat/bin/python
cd /mnt/Data1/Peoples/faiz836b/DNP-3DDMR-IR/TherMAM-NeRF || exit 1

OUT="results/cohort_calibBC_gpu${GPU}.csv"
LOG="calibBC_gpu${GPU}.log"

echo "GPU=${GPU}  offset=${OFFSET}  subset=${SUBSET}  h=${H_CONV}  T_air=${T_AIR}"
echo "output -> ${OUT}   log -> ${LOG}"

CUDA_VISIBLE_DEVICES="${GPU}" $PYTHON -u mukhmetov_recover.py --real \
    --subset "${SUBSET}" --stride 2 --offset "${OFFSET}" \
    --h-conv "${H_CONV}" --t-air "${T_AIR}" \
    --out "${OUT}" --resume 2>&1 | tee "${LOG}"
