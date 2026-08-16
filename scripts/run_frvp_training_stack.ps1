param(
    [string]$PythonExe = "",
    [string]$PreparedRoot = "artifacts\frvp_es_primary_current\phase04\prepared",
    [string]$XgbOutputRoot = "models\frvp_es_primary_xgb_v1",
    [string]$TcnReversalOutputRoot = "models\frvp_es_primary_tcn_reversal_v1",
    [string]$TcnContinuationOutputRoot = "models\frvp_es_primary_tcn_continuation_v1",
    [int]$XgbTrials = 20,
    [int]$TcnTrials = 12,
    [switch]$SkipXgboostTraining,
    [switch]$SkipTcnAttribution,
    [switch]$SkipTcnTraining
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$venvPython = Join-Path $repoRoot "ote_venv\Scripts\python.exe"

if ([string]::IsNullOrWhiteSpace($PythonExe)) {
    if (Test-Path $venvPython) {
        $PythonExe = $venvPython
    }
    else {
        $PythonExe = "python"
    }
}

function Invoke-Step {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name,
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    Write-Host ""
    Write-Host "==================================================" -ForegroundColor Cyan
    Write-Host $Name -ForegroundColor Cyan
    Write-Host "==================================================" -ForegroundColor Cyan
    Write-Host ("Command: " + $PythonExe + " " + ($Arguments -join " "))
    & $PythonExe @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE."
    }
}

Write-Host "==================================================" -ForegroundColor Green
Write-Host "FRVP Training Stack" -ForegroundColor Green
Write-Host "Working directory: $repoRoot"
Write-Host "Python:            $PythonExe"
Write-Host "Prepared root:     $PreparedRoot"
Write-Host "XGBoost output:    $XgbOutputRoot"
Write-Host "TCN reversal out:  $TcnReversalOutputRoot"
Write-Host "TCN continuation:  $TcnContinuationOutputRoot"
Write-Host "==================================================" -ForegroundColor Green

$xgbArgs = @(
    "-m",
    "model_training.ote_training.ote_xgboost_pipeline",
    "--prepared-root", $PreparedRoot,
    "--output-root", $XgbOutputRoot,
    "--backend", "xgboost",
    "--trials", $XgbTrials.ToString(),
    "--max-loaded-features", "160",
    "--window-min", "8",
    "--window-max", "40",
    "--event-tolerance-bars", "2",
    "--event-cooldown-bars", "4",
    "--calibration-method", "platt",
    "--cv-initial-train-rows", "900",
    "--cv-val-rows", "300",
    "--cv-step-rows", "300",
    "--cv-min-folds", "2",
    "--min-train-positive-rows", "350",
    "--min-val-positive-rows", "100",
    "--min-val-true-events", "20",
    "--seed", "42"
)

$tcnAttributionArgs = @(
    "-m",
    "preprocessing",
    "backend-attribution",
    "--prepared-root", $PreparedRoot,
    "--backend", "tcn",
    "--max-features", "160",
    "--attribution-max-rows", "100000",
    "--base-weight", "0.20",
    "--shap-weight", "0.55",
    "--shap-positive-weight", "0.25",
    "--attribution-floor-fraction", "0.15",
    "--attribution-cumulative-importance", "0.90",
    "--window-size", "24",
    "--batch-size", "256",
    "--attribution-batch-size", "128",
    "--torch-epochs", "10",
    "--hidden-size", "64",
    "--num-layers", "2",
    "--dropout", "0.20",
    "--torch-learning-rate", "0.001",
    "--weight-decay", "0.0001",
    "--gradient-clip", "1.0",
    "--top-n-features", "25",
    "--random-seed", "42"
)

$tcnReversalArgs = @(
    "-m",
    "model_training.ote_training.ote_xgboost_pipeline",
    "--prepared-root", $PreparedRoot,
    "--output-root", $TcnReversalOutputRoot,
    "--backend", "torch",
    "--model-type", "tcn",
    "--targets", "long_frvp_reversal", "short_frvp_reversal",
    "--trials", $TcnTrials.ToString(),
    "--max-loaded-features", "96",
    "--window-min", "16",
    "--window-max", "32",
    "--epochs", "18",
    "--batch-size", "256",
    "--calibration-method", "platt",
    "--cv-initial-train-rows", "900",
    "--cv-val-rows", "300",
    "--cv-step-rows", "300",
    "--cv-min-folds", "2",
    "--min-train-positive-rows", "350",
    "--min-val-positive-rows", "100",
    "--min-val-true-events", "20",
    "--seed", "42"
)

$tcnContinuationArgs = @(
    "-m",
    "model_training.ote_training.ote_xgboost_pipeline",
    "--prepared-root", $PreparedRoot,
    "--output-root", $TcnContinuationOutputRoot,
    "--backend", "torch",
    "--model-type", "tcn",
    "--targets", "long_frvp_continuation", "short_frvp_continuation",
    "--trials", $TcnTrials.ToString(),
    "--max-loaded-features", "96",
    "--window-min", "16",
    "--window-max", "32",
    "--epochs", "18",
    "--batch-size", "256",
    "--calibration-method", "platt",
    "--cv-initial-train-rows", "1800",
    "--cv-val-rows", "500",
    "--cv-step-rows", "500",
    "--cv-min-folds", "2",
    "--min-train-positive-rows", "700",
    "--min-val-positive-rows", "180",
    "--min-val-true-events", "35",
    "--seed", "42"
)

Push-Location $repoRoot
try {
    if (-not $SkipXgboostTraining) {
        Invoke-Step -Name "Step 1: XGBoost training" -Arguments $xgbArgs
    }

    if (-not $SkipTcnAttribution) {
        Invoke-Step -Name "Step 2: Standalone TCN backend attribution" -Arguments $tcnAttributionArgs
    }

    if (-not $SkipTcnTraining) {
        Invoke-Step -Name "Step 3: TCN reversal training" -Arguments $tcnReversalArgs
        Invoke-Step -Name "Step 4: TCN continuation training" -Arguments $tcnContinuationArgs
    }
}
finally {
    Pop-Location
}
