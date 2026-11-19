# Revision experiments — Access-2026-47475

This folder contains every analysis needed to answer the reviewers of manuscript
Access-2026-47475. The original pipeline (`main.py`, `src/`) is left unchanged for reference.
The mapping from each reviewer concern to the script and output file that answers it is given in
[`REVIEWER_OUTPUT_MAP.md`](REVIEWER_OUTPUT_MAP.md).

## 0. Division of work (important)

| Where | What |
|---|---|
| **Training workstation** (4 × RTX 3090) | **Only** runs the code in this folder (`run_all.bat` / `run_all.sh`) and produces the result files. |
| **Author's computer** (after training) | Results are copied back. **Editing the manuscript** (highlighted and clean versions) and **writing the response to reviewers** are done there, **not on the training workstation**. |

Nothing on the workstation edits the manuscript or the response letter; no Word documents are needed there.

## 1. Files to copy to the workstation

Copy the whole `PCE-son` folder. At minimum:

```
PCE-son/
├── data/dataset.csv
├── revision/            ← this folder (results_smoke/ is not needed)
├── requirements.txt, requirements-train.txt
├── app.py, templates/, static/, models/   (GUI; optional on the workstation)
```

## 2. Installation (once)

```bash
conda create -n pce python=3.10 -y
conda activate pce
cd PCE-son/revision
pip install -r requirements-revision.txt
```

Checks:

```bash
nvidia-smi                    # the GPUs must be listed
python bench_devices.py       # CPU vs GPU speed per model; prints the recommended --gpu-models
```

The dataset is small (616 rows), so a GPU can be **slower** than one CPU thread for some models.
Copy the recommendation of `bench_devices.py` into `GPU_MODELS` in `run_all.bat` / `run_all.sh`.
LightGBM, AdaBoost and MLP always run on CPU cores.

## 3. Running

**Windows:** edit the settings at the top of `run_all.bat`, then:

```bat
cd PCE-son\revision
run_all.bat
```

**Linux:**

```bash
cd PCE-son/revision
GPUS=0,1,2,3 WORKERS_PER_GPU=2 bash run_all.sh
```

| Setting | Meaning |
|---|---|
| `GPUS` | GPU indices to use: `0,1,2,3`, `0,2`, or `none` (CPU only). **Only the listed GPUs are used.** |
| `WORKERS_PER_GPU` | Concurrent jobs per GPU (default 2). |
| `CPU_WORKERS` | CPU workers; `-1` = all remaining cores. |
| `GPU_MODELS` | Models that run on the GPU slots (`XGBoost,CatBoost`). |

**No NVLink is required.** Every job uses exactly one GPU (`CUDA_VISIBLE_DEVICES` is set per worker
process); the GPUs work on independent jobs in parallel.

Manual control:

```bash
python run_experiments.py run --profile full --gpus 0,1,2,3 --workers-per-gpu 2 --cpu-workers -1
python run_experiments.py status --profile full                       # progress
python run_experiments.py run --profile full --gpus 0,1 --blocks final,rkf_main,lopo   # subset first
```

## 4. Power cuts, reboots, crashes

**Just run the same command again.**

- Every finished job is written **atomically** to `results_full/budget_*/jobs/<job>.json`
  (temporary file → fsync → rename): a power cut never leaves a half-written result.
- **Every objective evaluation** of a tuning run is appended to `cache/<job>.jsonl`. An interrupted
  EO / random-search / TPE run is replayed with the same seed; stored evaluations are returned from
  the cache, so the search **continues from the step where it stopped** and gives exactly the same
  result as an uninterrupted run (verified by `tests/test_resume.py`).
- Stale claims are cleared automatically and failed jobs are retried (2 retry rounds per run).
- A heartbeat lock prevents two runners from writing to the same folder; after a reboot the lock of
  the dead process is detected and ignored.
- `04_hyperparam_sensitivity.py` caches every grid point in the same way.

**Automatic restart (recommended):**

- Windows: `powershell -ExecutionPolicy Bypass -File install_autostart_windows.ps1` registers a
  scheduled task that starts `run_all.bat` at logon. Enable automatic logon and set
  **"Restore on AC Power Loss = Power On"** in the BIOS.
- Linux: add to `crontab -e`:
  `@reboot cd /path/PCE-son/revision && GPUS=0,1,2,3 bash run_all.sh >> autostart.log 2>&1`

## 5. Experiment blocks (profile `full`, 1959 jobs, ≈140 CPU-hours single-threaded)

| Block | Content |
|---|---|
| `final` | EO tuning on all data + final models (complete Table 4, SHAP, GUI) |
| `rkf_main` | Nested 3 × 5-fold CV, EO and default hyper-parameters |
| `rkf_tuners` | Random search and Bayesian (TPE) with the **same evaluation budget** as EO |
| `lopo` | Leave-one-PCE-group-out (near-duplicate pairs in the same group) |
| `lofa` | Leave-one-fly-ash-level-out |
| `nodup_rkf`, `nodup_lopo` | Near-duplicate admixtures removed (Y1) |
| `base_mixonly`, `base_mixonly_lopo` | Baseline without admixture information (X19–X21 only) |
| `base_pceid` | Admixture identity (one-hot) instead of the descriptors |
| `legacy_mice` | Original global-MICE procedure (targets imputed), scored on measured vs imputed rows |
| `sens` | EO population {10, 20, 30} × iterations {50, 100} × 5 seeds |

Budget: EO 20 agents × 100 iterations = 2120 objective evaluations per job; inner CV 3-fold.
Preprocessing (zero-variance removal, standardisation, mRMR) and tuning are fitted **only on the
training part of each outer fold**.

## 6. Outputs to bring back

| Folder | Content |
|---|---|
| `results_audit/` | Descriptive statistics, missingness, duplicates, correlation matrix, mRMR scores, flowchart |
| `results_full/budget_*/analysis/` | Tables 4–6, model and tuner tests, LOPO, LOFA, baselines, legacy MICE, residuals, EO sensitivity, figures, `summary.md` |
| `results_full/budget_*/final_analysis/` | SHAP tables, interactions and 600-dpi figures, learning curves |
| `results_full/budget_*/hyperparam_sensitivity/` | One-at-a-time hyper-parameter sensitivity |
| `results_full/budget_*/final_models/`, `../models_v2/` | Final pipelines; the GUI loads `models_v2/` automatically |
| `results_full/budget_*/jobs.zip`, `cache.zip` | Result of every job (with its seed) and every objective evaluation of every tuning run. Archived to keep the repository small: unzip both in `results_full/budget_c73d2068/` before re-running `02_aggregate.py` or resuming the run |

The `cache/` folder can be large and does not need to be copied back.

## 7. Data corrections (`rev/data.py`)

1. All 154 rows with X20 = "-" are mixtures without fly ash (UK0); X20 is set to **0**. No input
   variable then has missing values, so no imputation is used.
2. **No target imputation**: Y2 models use only the 252 measured rows.
3. X21 = PCE dosage (g per 700 g binder, 0.5–2.0 %); X22 = mixing water (245 g, constant).
4. Near-duplicate admixtures (K/K3:1, S5/S5-3:1, F9/F9-3:1) share one group in grouped validation.
5. S-series X13/X15 values are kept as reported (confirmed by the authors). Y2 unit: cm.

## 8. Smoke test (already done on the author's computer)

`python run_experiments.py run --profile smoke --gpus none --cpu-workers 6` exercises every code path
with a tiny budget. Smoke results are **not** for the manuscript.
