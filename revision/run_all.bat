@echo off
REM ===================================================================
REM  PCE revision - complete pipeline (Windows).  Safe to re-run at any time:
REM  finished jobs are skipped and interrupted jobs resume.
REM  Edit the settings below for the workstation.
REM ===================================================================
setlocal
set CONDA_ENV=pce
set PROFILE=full
REM GPUs to use (comma separated, e.g. 0,1,2,3 or 0,2) - "none" = CPU only
set GPUS=none
set WORKERS_PER_GPU=2
REM -1 = all remaining CPU cores
set CPU_WORKERS=32
REM result of bench_devices.py
set GPU_MODELS=XGBoost,CatBoost

cd /d "%~dp0"
call conda activate %CONDA_ENV% || (echo conda env %CONDA_ENV% not found & exit /b 1)

REM keep the machine awake while running (AC power)
powercfg /change standby-timeout-ac 0 >nul 2>&1
powercfg /change hibernate-timeout-ac 0 >nul 2>&1

python 00_data_audit.py || exit /b 1
python 01_mrmr_scores.py %PROFILE% || exit /b 1
python 05_flowchart.py || exit /b 1
python run_experiments.py run --profile %PROFILE% --gpus %GPUS% --workers-per-gpu %WORKERS_PER_GPU% --cpu-workers %CPU_WORKERS% --gpu-models %GPU_MODELS% || exit /b 1
python 02_aggregate.py --profile %PROFILE% || exit /b 1
python 03_final_analysis.py --profile %PROFILE% || exit /b 1
python 04_hyperparam_sensitivity.py --profile %PROFILE% || exit /b 1
echo.
echo ALL DONE. Results: %~dp0results_%PROFILE%
endlocal
