@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>nul
if errorlevel 1 (
  echo Python Launcher not found. Install Python 3.9+ and enable the py launcher.
  pause
  exit /b 1
)
if not exist .venv py -3 -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -e .
if errorlevel 1 (echo Dependency installation failed.& pause & exit /b 1)
python -m bgrfmvc.experiments.run_unsupervised --dataset demo --datasets-config configs/demo.yaml --method-config configs/default.yaml --output-dir results/demo --seed 0
if errorlevel 1 (echo BGRF-MVC run failed.& pause & exit /b 1)
echo Finished. Results are in results\demo
pause
