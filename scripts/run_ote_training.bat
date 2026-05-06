@echo off
setlocal

rem Run from the repository root no matter where the script is launched from.
cd /d "%~dp0\.."

set "VENV_PYTHON=%CD%\ote_venv\Scripts\python.exe"
set "PYTHON_CMD=python"
if not defined PREPARED_ROOT set "PREPARED_ROOT=data/prepared/eurusd_5min_ote_full"
if not defined OUTPUT_ROOT set "OUTPUT_ROOT=models/ote_multi_family_tcn_v1"
if not defined BACKEND set "BACKEND=torch"
if not defined MODEL_TYPE set "MODEL_TYPE=tcn"

if not defined VIRTUAL_ENV if exist "%VENV_PYTHON%" (
    set "PYTHON_CMD=%VENV_PYTHON%"
)

echo ==================================================
echo OTE Multi-Family Training
echo Working directory: %CD%
echo Python: %PYTHON_CMD%
echo Prepared root: %PREPARED_ROOT%
echo Output root: %OUTPUT_ROOT%
echo Backend: %BACKEND%
echo Model type: %MODEL_TYPE%
echo ==================================================
echo.
echo Supported prepared targets after rerunning labeling/features/preprocessing:
echo   long_reversal short_reversal long_reversal_entry short_reversal_entry
echo   long_continuation_pullback short_continuation_pullback
echo   long_continuation_entry short_continuation_entry
echo   long_breakout short_breakout long_breakout_entry short_breakout_entry
echo.

if "%~1"=="" (
    echo No explicit targets supplied. Auto-discovering all prepared target directories...
    "%PYTHON_CMD%" -m model_training.ote_training.ote_xgboost_pipeline --prepared-root "%PREPARED_ROOT%" --output-root "%OUTPUT_ROOT%" --backend %BACKEND% --model-type %MODEL_TYPE%
) else (
    echo Training requested target(s): %*
    "%PYTHON_CMD%" -m model_training.ote_training.ote_xgboost_pipeline --prepared-root "%PREPARED_ROOT%" --output-root "%OUTPUT_ROOT%" --backend %BACKEND% --model-type %MODEL_TYPE% --targets %*
)

exit /b %errorlevel%
