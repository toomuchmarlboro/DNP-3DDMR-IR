#!/bin/bash
# ---------------------------------------------------------------------------
# Re-run the FULL 122-patient cohort with the CORRECTED chest-wall BC
# (normal-based posterior surface = body side; fixed 2026-06-17). All other
# settings = production (h=10, T_air=20, 3 mm mesh). Output to NEW CSVs so the
# old (buggy) results stay for before/after comparison.
#
# Usage (two tmux panes):
#   pane 0:  ./run_BCfix.sh 0 0
#   pane 1:  ./run_BCfix.sh 1 1
#
# No --subset => all complete patients; stride 2 splits 61/61 across GPUs.
# ---------------------------------------------------------------------------
set -u
GPU="${1:-0}"
OFFSET="${2:-0}"
PYTHON=/mnt/Data1/Peoples/faiz836b/miniconda3/envs/bioheat/bin/python
cd /mnt/Data1/Peoples/faiz836b/DNP-3DDMR-IR/TherMAM-NeRF || exit 1

OUT="results/cohort_BCfix_gpu${GPU}.csv"
LOG="BCfix_gpu${GPU}.log"
echo "GPU=${GPU} offset=${OFFSET}  CORRECTED chest-wall BC (normal-based)"
echo "output -> ${OUT}   log -> ${LOG}"

CUDA_VISIBLE_DEVICES="${GPU}" $PYTHON -u mukhmetov_recover.py --real \
    --stride 2 --offset "${OFFSET}" \
    --out "${OUT}" --resume 2>&1 | tee "${LOG}"