param(
    [string]$PythonExe = "",
    [string]$RegistryPath = "models\ict_es_primary_model_registry_current.json",
    [string]$RegimeOutputRoot = "",
    [string]$ThresholdOutputRoot = "",
    [string]$BacktestOutputRoot = "",
    [int]$BootstrapIterations = 200,
    [int]$MinPositiveEvents = 20,
    [double]$MinEventsPerMonth = 1.0,
    [double]$MinTradesPerWeek = 3.0,
    [ValidateSet("promotion_quality", "research")]
    [string]$EvaluationContractMode = "promotion_quality",
    [double]$PromotionMinTradesPerWeekFloor = 3.0,
    [int]$BacktestMinTrainYears = 2,
    [double]$BacktestMaxTrainYears = 0,
    [int]$BacktestTestWindowMonths = 3,
    [int]$BacktestRollingStepMonths = 3,
    [string]$BacktestMinScheduledTestStart = "",
    [int]$BacktestMinFolds = 7,
    [int]$BacktestMaxFolds = 0,
    [double]$MinimumSharpe = 0.8,
    [double]$MinimumDsr = 0.3,
    [string[]]$ModelRoots = @(),
    [string[]]$ModelIds = @(),
    [string]$TargetedFilterPreset = "",
    [switch]$RebuildRegistry,
    [switch]$SkipRegime,
    [switch]$SkipThreshold,
    [switch]$SkipBacktest
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

if ([string]::IsNullOrWhiteSpace($RegimeOutputRoot)) {
    $RegimeOutputRoot = "model_testing\reports\ict_regime_slices\ict_es_primary_$runStamp"
}
if ([string]::IsNullOrWhiteSpace($ThresholdOutputRoot)) {
    $ThresholdOutputRoot = "model_testing\reports\ict_threshold_policies\ict_es_primary_$runStamp"
}
if ([string]::IsNullOrWhiteSpace($BacktestOutputRoot)) {
    $BacktestOutputRoot = "model_testing\reports\ict_backtests\ict_es_primary_$runStamp"
}

if (-not $PSBoundParameters.ContainsKey("MinTradesPerWeek")) {
    if ($EvaluationContractMode -eq "research") {
        $MinTradesPerWeek = 0.5
    }
    else {
        $MinTradesPerWeek = $PromotionMinTradesPerWeekFloor
    }
}

if (
    ($EvaluationContractMode -eq "promotion_quality") `
    -and ($MinTradesPerWeek -lt $PromotionMinTradesPerWeekFloor)
) {
    throw "Promotion-quality ICT evaluation requires MinTradesPerWeek >= $PromotionMinTradesPerWeekFloor. Use -EvaluationContractMode research for low-frequency exploratory runs."
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

function Ensure-Registry {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RegistryPath,
        [string[]]$ModelRoots = @(),
        [switch]$Force
    )

    if ((Test-Path $RegistryPath) -and (-not $Force)) {
        return
    }

    if ($Force -and (Test-Path $RegistryPath)) {
        Write-Host ""
        Write-Host "Rebuilding ICT registry at $RegistryPath" -ForegroundColor Yellow
    }
    elseif (-not (Test-Path $RegistryPath)) {
        Write-Host ""
        Write-Host "Registry not found at $RegistryPath" -ForegroundColor Yellow
        Write-Host "Auto-generating ICT candidate registry." -ForegroundColor Yellow
    }

    $builderArgs = @(
        "scripts/build_ict_candidate_registry.py"
    )
    foreach ($modelRoot in @($ModelRoots)) {
        if (-not [string]::IsNullOrWhiteSpace($modelRoot)) {
            $builderArgs += @("--model-root", $modelRoot)
        }
    }
    $builderArgs += @(
        "--source-registry-path", $RegistryPath,
        "--output-path", $RegistryPath
    )

    Invoke-Step -Name "Bootstrap: build ICT registry" -Arguments $builderArgs
}

function Resolve-BacktestFoldContract {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RegimeReportRoot,
        [Parameter(Mandatory = $true)]
        [string]$RegistryPath,
        [Parameter(Mandatory = $true)]
        [int]$RequestedMinFolds,
        [Parameter(Mandatory = $true)]
        [string]$EvaluationContractMode,
        [int]$MinTrainYears,
        [double]$MaxTrainYears,
        [int]$TestWindowMonths,
        [int]$RollingStepMonths,
        [string]$MinScheduledTestStart,
        [int]$MaxFolds,
        [string[]]$ModelIds = @()
    )

    $auditArgs = @(
        "scripts/audit_ote_backtest_folds.py",
        "--regime-report-root", $RegimeReportRoot,
        "--registry-path", $RegistryPath,
        "--status", "candidate",
        "--min-train-years", "$MinTrainYears",
        "--test-window-months", "$TestWindowMonths",
        "--rolling-step-months", "$RollingStepMonths",
        "--requested-min-folds", "$RequestedMinFolds"
    )

    if ($MaxTrainYears -gt 0) {
        $auditArgs += @("--max-train-years", "$MaxTrainYears")
    }
    if (-not [string]::IsNullOrWhiteSpace($MinScheduledTestStart)) {
        $auditArgs += @("--min-scheduled-test-start", $MinScheduledTestStart)
    }
    if ($MaxFolds -gt 0) {
        $auditArgs += @("--max-folds", "$MaxFolds")
    }
    foreach ($modelId in @($ModelIds)) {
        $auditArgs += @("--model-id", $modelId)
    }

    $auditJson = & $PythonExe @auditArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Backtest fold audit failed with exit code $LASTEXITCODE."
    }
    $audit = $auditJson | Out-String | ConvertFrom-Json
    $availableMinFolds = [int]$audit.min_available_fold_count
    if ($RequestedMinFolds -le $availableMinFolds) {
        return [PSCustomObject]@{
            requested_min_folds    = $RequestedMinFolds
            available_min_folds    = $availableMinFolds
            effective_min_folds    = $RequestedMinFolds
            used_auto_relaxed_folds = $false
            insufficient_models    = @($audit.insufficient_models)
        }
    }

    $insufficientModels = @($audit.insufficient_models)
    if ($insufficientModels.Count -eq 0) {
        return [PSCustomObject]@{
            requested_min_folds    = $RequestedMinFolds
            available_min_folds    = $availableMinFolds
            effective_min_folds    = $availableMinFolds
            used_auto_relaxed_folds = $RequestedMinFolds -gt $availableMinFolds
            insufficient_models    = @()
        }
    }

    if ($EvaluationContractMode -eq "promotion_quality") {
        Write-Host ""
        Write-Host "Backtest fold audit failed the promotion-quality contract." -ForegroundColor Red
        Write-Host "Requested min folds: $RequestedMinFolds" -ForegroundColor Red
        Write-Host "Available min folds: $availableMinFolds" -ForegroundColor Red
        Write-Host "Models below the requested fold count:" -ForegroundColor Red
        foreach ($modelId in $insufficientModels) {
            Write-Host "  - $modelId" -ForegroundColor Red
        }
        throw "Promotion-quality ICT evaluation cannot auto-relax min folds."
    }

    Write-Host ""
    Write-Host "Backtest fold audit detected a stricter gate than the current ICT predictions can support." -ForegroundColor Yellow
    Write-Host "Auto-relaxing min folds from $RequestedMinFolds to $availableMinFolds for this run." -ForegroundColor Yellow
    Write-Host "Models below the requested fold count:" -ForegroundColor Yellow
    foreach ($modelId in $insufficientModels) {
        Write-Host "  - $modelId" -ForegroundColor Yellow
    }
    return [PSCustomObject]@{
        requested_min_folds    = $RequestedMinFolds
        available_min_folds    = $availableMinFolds
        effective_min_folds    = $availableMinFolds
        used_auto_relaxed_folds = $true
        insufficient_models    = $insufficientModels
    }
}

Write-Host "==================================================" -ForegroundColor Green
Write-Host "ICT ES Post-Training Evaluation" -ForegroundColor Green
Write-Host "Working directory:     $repoRoot"
Write-Host "Python:                $PythonExe"
Write-Host "Registry:              $RegistryPath"
Write-Host "Regime root:           $RegimeOutputRoot"
Write-Host "Threshold root:        $ThresholdOutputRoot"
Write-Host "Backtest root:         $BacktestOutputRoot"
Write-Host "Eval contract mode:    $EvaluationContractMode"
Write-Host "Min positive events:   $MinPositiveEvents"
Write-Host "Min events/month:      $MinEventsPerMonth"
Write-Host "Min trades/week:       $MinTradesPerWeek"
Write-Host "Promotion trade floor: $PromotionMinTradesPerWeekFloor"
Write-Host "Backtest min yrs:      $BacktestMinTrainYears"
if ($BacktestMaxTrainYears -gt 0) {
    Write-Host "Backtest max yrs:      $BacktestMaxTrainYears"
}
if (-not [string]::IsNullOrWhiteSpace($BacktestMinScheduledTestStart)) {
    Write-Host "Backtest min test:     $BacktestMinScheduledTestStart"
}
if ($BacktestMaxFolds -gt 0) {
    Write-Host "Backtest max folds:    $BacktestMaxFolds"
}
Write-Host "Minimum Sharpe:        $MinimumSharpe"
Write-Host "Minimum DSR:           $MinimumDsr"
Write-Host "==================================================" -ForegroundColor Green

$ModelRoots = @(Expand-DelimitedValues -Values $ModelRoots)
$ModelIds = @(Expand-DelimitedValues -Values $ModelIds)

$regimeArgs = @(
    "scripts/run_ote_regime_slice_report.py",
    "--registry-path", $RegistryPath,
    "--output-root", $RegimeOutputRoot,
    "--status", "candidate",
    "--bootstrap-iterations", "$BootstrapIterations",
    "--min-positive-events", "$MinPositiveEvents"
)

$thresholdArgs = @(
    "scripts/run_ote_threshold_policy_search.py",
    "--regime-report-root", $RegimeOutputRoot,
    "--registry-path", $RegistryPath,
    "--output-root", $ThresholdOutputRoot,
    "--status", "candidate",
    "--instrument", "es",
    "--min-positive-events", "$MinPositiveEvents",
    "--min-events-per-month", "$MinEventsPerMonth",
    "--min-trades-per-week", "$MinTradesPerWeek",
    "--evaluation-contract-mode", $EvaluationContractMode,
    "--promotion-min-trades-per-week-floor", "$PromotionMinTradesPerWeekFloor",
    "--write-policy-decisions"
)

if ($ModelIds.Count -gt 0) {
    foreach ($modelId in $ModelIds) {
        $regimeArgs += @("--model-id", $modelId)
        $thresholdArgs += @("--model-id", $modelId)
    }
}

if (-not [string]::IsNullOrWhiteSpace($TargetedFilterPreset)) {
    $thresholdArgs += @("--targeted-filter-preset", $TargetedFilterPreset)
}

Push-Location $repoRoot
try {
    Ensure-Registry -RegistryPath $RegistryPath -ModelRoots $ModelRoots -Force:$RebuildRegistry

    if (-not $SkipRegime) {
        Invoke-Step -Name "Step 1: ICT regime-slice report" -Arguments $regimeArgs
    }

    if (-not $SkipThreshold) {
        Invoke-Step -Name "Step 2: ICT threshold policy search" -Arguments $thresholdArgs
    }

    if (-not $SkipBacktest) {
        $foldContract = Resolve-BacktestFoldContract `
            -RegimeReportRoot $RegimeOutputRoot `
            -RegistryPath $RegistryPath `
            -RequestedMinFolds $BacktestMinFolds `
            -EvaluationContractMode $EvaluationContractMode `
            -MinTrainYears $BacktestMinTrainYears `
            -MaxTrainYears $BacktestMaxTrainYears `
            -TestWindowMonths $BacktestTestWindowMonths `
            -RollingStepMonths $BacktestRollingStepMonths `
            -MinScheduledTestStart $BacktestMinScheduledTestStart `
            -MaxFolds $BacktestMaxFolds `
            -ModelIds $ModelIds

        $backtestArgs = @(
            "scripts/run_ote_policy_backtest.py",
            "--regime-report-root", $RegimeOutputRoot,
            "--registry-path", $RegistryPath,
            "--output-root", $BacktestOutputRoot,
            "--status", "candidate",
            "--instrument", "es",
            "--min-train-years", "$BacktestMinTrainYears",
            "--test-window-months", "$BacktestTestWindowMonths",
            "--rolling-step-months", "$BacktestRollingStepMonths",
            "--min-folds", "$($foldContract.effective_min_folds)",
            "--requested-min-folds", "$($foldContract.requested_min_folds)",
            "--available-min-folds", "$($foldContract.available_min_folds)",
            "--min-positive-events", "$MinPositiveEvents",
            "--min-events-per-month", "$MinEventsPerMonth",
            "--min-trades-per-week", "$MinTradesPerWeek",
            "--evaluation-contract-mode", $EvaluationContractMode,
            "--promotion-min-trades-per-week-floor", "$PromotionMinTradesPerWeekFloor",
            "--minimum-sharpe", "$MinimumSharpe",
            "--minimum-dsr", "$MinimumDsr"
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
                $backtestArgs += @("--model-id", $modelId)
            }
        }

        if (-not [string]::IsNullOrWhiteSpace($TargetedFilterPreset)) {
            $backtestArgs += @("--targeted-filter-preset", $TargetedFilterPreset)
        }

        Invoke-Step -Name "Step 3: ICT walk-forward policy backtest" -Arguments $backtestArgs
    }
}
finally {
    Pop-Location
}
