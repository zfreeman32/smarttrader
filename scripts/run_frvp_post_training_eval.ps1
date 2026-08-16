param(
    [string]$PythonExe = "",
    [string]$RegistryPath = "models\frvp_es_primary_model_registry_current.json",
    [string]$RegimeOutputRoot = "model_testing\reports\frvp_regime_slices\frvp_es_primary_current",
    [string]$ThresholdOutputRoot = "model_testing\reports\frvp_threshold_policies\frvp_es_primary_current",
    [string]$BacktestOutputRoot = "model_testing\reports\frvp_backtests\frvp_es_primary_current",
    [int]$BacktestMinTrainYears = 2,
    [double]$BacktestMaxTrainYears = 0,
    [int]$BacktestTestWindowMonths = 3,
    [int]$BacktestRollingStepMonths = 3,
    [string]$BacktestMinScheduledTestStart = "",
    [int]$BacktestMinFolds = 8,
    [int]$BacktestMaxFolds = 0,
    [string[]]$ModelRoots = @(),
    [string[]]$ModelIds = @(),
    [string]$TargetedFilterPreset = "",
    [string]$SpreadCostMode = "auto",
    [switch]$SkipRegime,
    [switch]$SkipThreshold,
    [switch]$SkipBacktest
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

function Expand-DelimitedValues {
    param(
        [string[]]$Values = @()
    )

    $expanded = @()
    foreach ($value in @($Values)) {
        if ([string]::IsNullOrWhiteSpace($value)) {
            continue
        }
        foreach ($part in ($value -split ",")) {
            $trimmed = $part.Trim()
            if (-not [string]::IsNullOrWhiteSpace($trimmed)) {
                $expanded += $trimmed
            }
        }
    }
    return @($expanded)
}

function Get-DefaultModelRoots {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RegistryPath
    )

    $currentRoots = @(
        "models\frvp_es_primary_xgb_v1",
        "models\frvp_es_primary_tcn_reversal_v1",
        "models\frvp_es_primary_tcn_continuation_v1"
    )

    $registryLeaf = Split-Path $RegistryPath -Leaf
    $match = [regex]::Match($registryLeaf, '^frvp_es_primary_model_registry_(.+)\.json$')
    if (-not $match.Success) {
        return $currentRoots
    }

    $suffix = $match.Groups[1].Value
    if ($suffix -eq "current") {
        return $currentRoots
    }

    return @(
        "models\frvp_es_primary_xgb_$suffix",
        "models\frvp_es_primary_tcn_reversal_$suffix",
        "models\frvp_es_primary_tcn_continuation_$suffix"
    )
}

function Ensure-Registry {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RegistryPath,
        [string[]]$ModelRoots = @()
    )

    if (Test-Path $RegistryPath) {
        return
    }

    $candidateRoots = @()
    if ($ModelRoots.Count -gt 0) {
        $candidateRoots = @($ModelRoots)
    }
    else {
        $candidateRoots = @(Get-DefaultModelRoots -RegistryPath $RegistryPath)
    }

    $existingRoots = @($candidateRoots | Where-Object { Test-Path $_ })
    if ($existingRoots.Count -eq 0) {
        throw "Registry '$RegistryPath' is missing and no FRVP model roots were found to rebuild it."
    }

    Write-Host ""
    Write-Host "Registry not found at $RegistryPath" -ForegroundColor Yellow
    Write-Host "Auto-generating FRVP candidate registry from:" -ForegroundColor Yellow
    foreach ($modelRoot in $existingRoots) {
        Write-Host "  - $modelRoot" -ForegroundColor Yellow
    }

    $builderArgs = @(
        "scripts/build_frvp_candidate_registry.py"
    )
    foreach ($modelRoot in $existingRoots) {
        $builderArgs += @("--model-root", $modelRoot)
    }
    $builderArgs += @(
        "--source-registry-path", "models\frvp_es_primary_model_registry_current.json",
        "--output-path", $RegistryPath
    )

    Invoke-Step -Name "Bootstrap: build missing FRVP registry" -Arguments $builderArgs
}

Write-Host "==================================================" -ForegroundColor Green
Write-Host "FRVP Post-Training Evaluation" -ForegroundColor Green
Write-Host "Working directory: $repoRoot"
Write-Host "Python:            $PythonExe"
Write-Host "Registry:          $RegistryPath"
Write-Host "Regime root:       $RegimeOutputRoot"
Write-Host "Threshold root:    $ThresholdOutputRoot"
Write-Host "Backtest root:     $BacktestOutputRoot"
Write-Host "Spread costs:      $SpreadCostMode"
Write-Host "Backtest min yrs:  $BacktestMinTrainYears"
if ($BacktestMaxTrainYears -gt 0) {
    Write-Host "Backtest max yrs:  $BacktestMaxTrainYears"
}
if (-not [string]::IsNullOrWhiteSpace($BacktestMinScheduledTestStart)) {
    Write-Host "Backtest min test: $BacktestMinScheduledTestStart"
}
if ($BacktestMaxFolds -gt 0) {
    Write-Host "Backtest max folds:$BacktestMaxFolds"
}
Write-Host "==================================================" -ForegroundColor Green

$ModelRoots = @(Expand-DelimitedValues -Values $ModelRoots)
$ModelIds = @(Expand-DelimitedValues -Values $ModelIds)

$regimeArgs = @(
    "scripts/run_ote_regime_slice_report.py",
    "--registry-path", $RegistryPath,
    "--output-root", $RegimeOutputRoot,
    "--status", "candidate",
    "--bootstrap-iterations", "200",
    "--min-positive-events", "50"
)

$thresholdArgs = @(
    "scripts/run_ote_threshold_policy_search.py",
    "--regime-report-root", $RegimeOutputRoot,
    "--registry-path", $RegistryPath,
    "--output-root", $ThresholdOutputRoot,
    "--status", "candidate",
    "--instrument", "es",
    "--spread-cost-mode", $SpreadCostMode,
    "--min-positive-events", "50",
    "--min-events-per-month", "3.0",
    "--min-trades-per-week", "3.0",
    "--write-policy-decisions"
)

$backtestArgs = @(
    "scripts/run_ote_policy_backtest.py",
    "--regime-report-root", $RegimeOutputRoot,
    "--registry-path", $RegistryPath,
    "--output-root", $BacktestOutputRoot,
    "--status", "candidate",
    "--instrument", "es",
    "--spread-cost-mode", $SpreadCostMode,
    "--min-train-years", "$BacktestMinTrainYears",
    "--test-window-months", "$BacktestTestWindowMonths",
    "--rolling-step-months", "$BacktestRollingStepMonths",
    "--min-folds", "$BacktestMinFolds",
    "--min-positive-events", "50",
    "--min-events-per-month", "3.0",
    "--min-trades-per-week", "3.0",
    "--minimum-sharpe", "0.8",
    "--maximum-drawdown-pct", "12.0",
    "--minimum-dsr", "0.3"
)

if ($BacktestMaxTrainYears -gt 0) {
    $backtestArgs += @("--max-train-years", "$BacktestMaxTrainYears")
}

if (-not [string]::IsNullOrWhiteSpace($BacktestMinScheduledTestStart)) {
    $backtestArgs += @("--min-scheduled-test-start", $BacktestMinScheduledTestStart)
}

if ($BacktestMaxFolds -gt 0) {
    $backtestArgs += @("--max-folds", "$BacktestMaxFolds")
}

if ($ModelIds.Count -gt 0) {
    foreach ($modelId in $ModelIds) {
        $regimeArgs += @("--model-id", $modelId)
        $thresholdArgs += @("--model-id", $modelId)
        $backtestArgs += @("--model-id", $modelId)
    }
}

if (-not [string]::IsNullOrWhiteSpace($TargetedFilterPreset)) {
    $thresholdArgs += @("--targeted-filter-preset", $TargetedFilterPreset)
    $backtestArgs += @("--targeted-filter-preset", $TargetedFilterPreset)
}

Push-Location $repoRoot
try {
    Ensure-Registry -RegistryPath $RegistryPath -ModelRoots $ModelRoots

    if (-not $SkipRegime) {
        Invoke-Step -Name "Step 1: FRVP regime-slice report" -Arguments $regimeArgs
    }

    if (-not $SkipThreshold) {
        Invoke-Step -Name "Step 2: FRVP threshold policy search" -Arguments $thresholdArgs
    }

    if (-not $SkipBacktest) {
        Invoke-Step -Name "Step 3: FRVP walk-forward policy backtest" -Arguments $backtestArgs
    }
}
finally {
    Pop-Location
}
