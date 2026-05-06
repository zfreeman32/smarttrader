@echo off
setlocal

rem Run from the repository root no matter where the script is launched from.
cd /d "%~dp0\.."

set "WAIT_SECONDS=120"
set "VENV_PYTHON=%CD%\ote_venv\Scripts\python.exe"
set "PYTHON_CMD=python"
set "RAW_SOURCE_TIMEZONE=GMT-6"
set "CANONICAL_TIMEZONE=UTC"
set "FEATURE_CLOCK_TIMEZONE=America/New_York"
set "MARKET_CLOSE_TIMEZONE=America/New_York"
set "BAR_TIMESTAMP_SEMANTICS=as_provided"
if not defined START_DATE set "START_DATE=2019-01-01"
set "LABEL_DATE_ARGS="
if defined START_DATE set "LABEL_DATE_ARGS=%LABEL_DATE_ARGS% --start-date %START_DATE%"
if defined END_DATE set "LABEL_DATE_ARGS=%LABEL_DATE_ARGS% --end-date %END_DATE%"

if not defined VIRTUAL_ENV if exist "%VENV_PYTHON%" (
    set "PYTHON_CMD=%VENV_PYTHON%"
)

echo ==================================================
echo EURUSD 5-Minute Multi-Family Label Pipeline
echo Working directory: %CD%
echo Python: %PYTHON_CMD%
echo Raw source timezone: %RAW_SOURCE_TIMEZONE%
echo Canonical timezone: %CANONICAL_TIMEZONE%
echo Feature clock timezone: %FEATURE_CLOCK_TIMEZONE%
echo Market close timezone: %MARKET_CLOSE_TIMEZONE%
echo Start date: %START_DATE%
if defined END_DATE (
echo End date: %END_DATE%
) else (
echo End date: latest bar in source
)
echo ==================================================
echo.

echo [1/3] Running labeling engine for reversal, continuation, and breakout labels...
"%PYTHON_CMD%" data/labeling/labeling_engine.py --input data/currency_data/eurusd-5m.csv --output data/labeling/labeled_data/eurusd_5min_ote_labels_full.csv --swings-output data/labeling/labeled_data/eurusd_5min_ote_swings_full.csv %LABEL_DATE_ARGS% --source-timezone %RAW_SOURCE_TIMEZONE% --canonical-timezone %CANONICAL_TIMEZONE% --bar-timestamp-semantics %BAR_TIMESTAMP_SEMANTICS%
if errorlevel 1 goto :failed

echo.
echo Waiting %WAIT_SECONDS% seconds before step 2...
timeout /t %WAIT_SECONDS% /nobreak
if errorlevel 1 goto :failed

echo.
echo [2/3] Building features...
"%PYTHON_CMD%" -m features.cli build data/labeling/labeled_data/eurusd_5min_ote_labels_full.csv --output data/features/eurusd_5min_ote_full.csv --recipe features/recipes/ote_extended.json --all-strategies --transform-workers 4 --optimize-memory --skip-strategy-errors --progress --strategy-timeout-seconds 600 --source-timezone %CANONICAL_TIMEZONE% --canonical-timezone %CANONICAL_TIMEZONE% --feature-clock-timezone %FEATURE_CLOCK_TIMEZONE% --market-close-timezone %MARKET_CLOSE_TIMEZONE%
if errorlevel 1 goto :failed

echo.
echo Waiting %WAIT_SECONDS% seconds before step 3...
timeout /t %WAIT_SECONDS% /nobreak
if errorlevel 1 goto :failed

echo.
echo [3/3] Preparing training-ready datasets for all discovered label targets...
"%PYTHON_CMD%" -m preprocessing prepare data/features/eurusd_5min_ote_full.csv --output-dir data/prepared/eurusd_5min_ote_full --scaler none --corr-threshold 0.98 --similarity-threshold 0.995 --max-analysis-rows 10000
if errorlevel 1 goto :failed

echo.
echo Pipeline completed successfully.
exit /b 0

:failed
echo.
echo Pipeline failed with exit code %errorlevel%.
exit /b %errorlevel%
