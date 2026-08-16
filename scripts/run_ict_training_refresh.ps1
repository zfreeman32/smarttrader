param(
    [string]$PythonExe = "",
    [string]$PreparedRoot = "artifacts\ict_es_primary_refresh_20260724_spacing_refit_final_confirm\phase04_prepared\prepared",
    [string]$OutputRoot = "",
    [string]$RegistryOutputPath = "",
    [string]$SourceRegistryPath = "models\ict_es_primary_model_registry_current.json",
    [string]$RegimeOutputRoot = "",
    [string]$ThresholdOutputRoot = "",
    [string]$BacktestOutputRoot = "",
    [string[]]$Targets = @(
        "long_ict_reversal",
        "short_ict_reversal",
        "long_ict_continuation",
        "short_ict_continuation",
        "long_ict_meta",
        "short_ict_meta"
    ),
    [int]$Trials = 10,
    [ValidateSet("off", "ict", "all")]
    [string]$SequentialBootstrapMode = "ict",
    [int]$SequentialBootstrapMaxCandidates = 0,
    [switch]$SkipPostEval
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$venvPython = Join-Path $repoRoot "ote_venv\Scripts\python.exe"
$runStamp = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssZ")

if ([string]::IsNullOrWhiteSpace($PythonExe)) {
    if (Test-Path $venvPython) {
        $PythonExe = $venvPython
    }
    else {
        $PythonExe = "python"
    }
}

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $OutputRoot = "models\ict_es_primary_xgb_bootstrap_$runStamp"
}
if ([string]::IsNullOrWhiteSpace($RegistryOutputPath)) {
    $RegistryOutputPath = "models\ict_es_primary_model_registry_bootstrap_$runStamp.json"
}
if ([string]::IsNullOrWhiteSpace($RegimeOutputRoot)) {
    $RegimeOutputRoot = "model_testing\reports\ict_regime_slices\ict_es_primary_bootstrap_$runStamp"
}
if ([string]::IsNullOrWhiteSpace($ThresholdOutputRoot)) {
    $ThresholdOutputRoot = "model_testing\reports\ict_threshold_policies\ict_es_primary_bootstrap_$runStamp"
}
if ([string]::IsNullOrWhiteSpace($BacktestOutputRoot)) {
    $BacktestOutputRoot = "model_testing\reports\ict_backtests\ict_es_primary_bootstrap_$runStamp"
}

function Invoke-Step {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name,
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    Write-Output ""
    Write-Output "=================================================="
    Write-Output $Name
    Write-Output "=================================================="
    Write-Output ("Command: " + $PythonExe + " " + ($Arguments -join " "))
    & $PythonExe @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE."
    }
}

$trainingArgs = @(
    "model_training/ote_training/ote_xgboost_pipeline.py",
    "--prepared-root", $PreparedRoot,
    "--output-root", $OutputRoot,
    "--backend", "xgboost",
    "--targets"
)
$trainingArgs += @($Targets)
$trainingArgs += @(
    "--trials", "$Trials",
    "--cv-initial-train-rows", "400",
    "--cv-val-rows", "128",
    "--cv-step-rows", "96",
    "--cv-min-folds", "2",
    "--min-train-positive-rows", "115",
    "--min-val-positive-rows", "19",
    "--min-val-true-events", "13",
    "--label-max-holding-bars", "20",
    "--label-exclusion-pre-bars", "10",
    "--label-zone-pre-bars", "2",
    "--purge-buffer-bars", "12",
    "--max-loaded-features", "96",
    "--top-feature-max", "96",
    "--window-min", "8",
    "--window-max", "24",
    "--event-tolerance-bars", "2",
    "--event-cooldown-bars", "4",
    "--calibration-method", "platt",
    "--seed", "42",
    "--sequential-bootstrap-mode", $SequentialBootstrapMode
)
if ($SequentialBootstrapMaxCandidates -gt 0) {
    $trainingArgs += @("--sequential-bootstrap-max-candidates", "$SequentialBootstrapMaxCandidates")
}

Invoke-Step -Name "ICT training refresh" -Arguments $trainingArgs

$registryArgs = @(
    "scripts/build_ict_candidate_registry.py",
    "--model-root", $OutputRoot,
    "--source-registry-path", $SourceRegistryPath,
    "--output-path", $RegistryOutputPath
)
Invoke-Step -Name "ICT registry rebuild" -Arguments $registryArgs

if (-not $SkipPostEval) {
    Write-Output ""
    Write-Output "=================================================="
    Write-Output "ICT post-training evaluation"
    Write-Output "=================================================="
    Write-Output (
        "Command: powershell -File scripts/run_ict_post_training_eval.ps1 -PythonExe " + $PythonExe +
        " -RegistryPath " + $RegistryOutputPath +
        " -RegimeOutputRoot " + $RegimeOutputRoot +
        " -ThresholdOutputRoot " + $ThresholdOutputRoot +
        " -BacktestOutputRoot " + $BacktestOutputRoot
    )
    & powershell -ExecutionPolicy Bypass -File (
        Join-Path $repoRoot "scripts\run_ict_post_training_eval.ps1"
    ) `
        -PythonExe $PythonExe `
        -RegistryPath $RegistryOutputPath `
        -RegimeOutputRoot $RegimeOutputRoot `
        -ThresholdOutputRoot $ThresholdOutputRoot `
        -BacktestOutputRoot $BacktestOutputRoot
    if ($LASTEXITCODE -ne 0) {
        throw "ICT post-training evaluation failed with exit code $LASTEXITCODE."
    }
}

Write-Output ""
Write-Output "Completed."
Write-Output ("output_root=" + $OutputRoot)
Write-Output ("registry_path=" + $RegistryOutputPath)
Write-Output ("regime_output_root=" + $RegimeOutputRoot)
Write-Output ("threshold_output_root=" + $ThresholdOutputRoot)
Write-Output ("backtest_output_root=" + $BacktestOutputRoot)
