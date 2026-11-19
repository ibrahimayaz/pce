#!/usr/bin/env bash
# ===================================================================
#  PCE revision - complete pipeline (Linux).  Safe to re-run at any time:
#  finished jobs are skipped and interrupted jobs resume.
# ===================================================================
set -euo pipefail
CONDA_ENV="${CONDA_ENV:-pce}"
PROFILE="${PROFILE:-full}"
GPUS="${GPUS:-0,1,2,3}"            # e.g. 0,1,2,3 | 0,2 | none
WORKERS_PER_GPU="${WORKERS_PER_GPU:-2}"
CPU_WORKERS="${CPU_WORKERS:--1}"   # -1 = all remaining cores
GPU_MODELS="${GPU_MODELS:-XGBoost,CatBoost}"

cd "$(dirname "$0")"
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

python 00_data_audit.py
python 01_mrmr_scores.py "$PROFILE"
python 05_flowchart.py
python run_experiments.py run --profile "$PROFILE" --gpus "$GPUS" --workers-per-gpu "$WORKERS_PER_GPU" \
       --cpu-workers "$CPU_WORKERS" --gpu-models "$GPU_MODELS"
python 02_aggregate.py --profile "$PROFILE"
python 03_final_analysis.py --profile "$PROFILE"
python 04_hyperparam_sensitivity.py --profile "$PROFILE"
echo "ALL DONE. Results: $(pwd)/results_${PROFILE}"
