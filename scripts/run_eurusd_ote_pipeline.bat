@echo off
setlocal

rem Run from the repository root no matter where the script is launched from.
cd /d "%~dp0\.."

set "WAIT_SECONDS=120"
set "VENV_PYTHON=%CD%\ote_venv\Scripts\python.exe"
set "PYTHON_CMD=python"

if not defined VIRTUAL_ENV if exist "%VENV_PYTHON%" (
    set "PYTHON_CMD=%VENV_PYTHON%"
)

echo ==================================================
echo EURUSD 5-Minute OTE Pipeline
echo Working directory: %CD%
echo Python: %PYTHON_CMD%
echo ==================================================
echo.

echo [1/3] Running labeling engine...
"%PYTHON_CMD%" data/labeling/labeling_engine.py --input data/currency_data/eurusd-5m.csv --output data/labeling/labeled_data/eurusd_5min_ote_labels_full.csv --swings-output data/labeling/labeled_data/eurusd_5min_ote_swings_full.csv
if errorlevel 1 goto :failed

echo.
echo Waiting %WAIT_SECONDS% seconds before step 2...
timeout /t %WAIT_SECONDS% /nobreak
if errorlevel 1 goto :failed

echo.
echo [2/3] Building features...
"%PYTHON_CMD%" -m features.cli build data/labeling/labeled_data/eurusd_5min_ote_labels_full.csv --output data/features/eurusd_5min_ote_full.csv --recipe features/recipes/ote_extended.json --all-strategies --transform-workers 4 --optimize-memory --skip-strategy-errors --progress --strategy-timeout-seconds 600
if errorlevel 1 goto :failed

echo.
echo Waiting %WAIT_SECONDS% seconds before step 3...
timeout /t %WAIT_SECONDS% /nobreak
if errorlevel 1 goto :failed

echo.
echo [3/3] Preprocessing features...
"%PYTHON_CMD%" -m features.cli preprocess data/features/eurusd_5min_ote_full.csv --output-dir data/prepared/eurusd_5min_ote_full --scaler none --corr-threshold 0.98 --similarity-threshold 0.995 --max-analysis-rows 10000
if errorlevel 1 goto :failed

echo.
echo Pipeline completed successfully.
exit /b 0

:failed
echo.
echo Pipeline failed with exit code %errorlevel%.
exit /b %errorlevel%
