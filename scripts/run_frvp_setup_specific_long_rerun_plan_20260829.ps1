param(
    [string]$PythonExe = "",
    [string]$LogFile = "frvp_setup_specific_long_rerun_20260829_opt_output.txt"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Continue"

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

$logPath = Join-Path $repoRoot $LogFile
$summary = New-Object System.Collections.Generic.List[object]

function Write-LogLine {
    param([string]$Message)

    Write-Host $Message
    Add-Content -Path $logPath -Value $Message -Encoding utf8
}

function Format-CommandLine {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Command,
        [string[]]$Arguments = @()
    )

    $quotedArgs = @()
    foreach ($argument in $Arguments) {
        if ($argument -match '\s') {
            $quotedArgs += '"' + $argument.Replace('"', '\"') + '"'
        }
        else {
            $quotedArgs += $argument
        }
    }
    return ($Command + " " + ($quotedArgs -join " ")).Trim()
}

function Invoke-LoggedStep {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name,
        [Parameter(Mandatory = $true)]
        [scriptblock]$Action,
        [Parameter(Mandatory = $true)]
        [string]$CommandLine,
        [switch]$UseLastExitCode
    )

    $started = Get-Date
    $exitCode = 0
    $script:StepExitCodeOverride = $null

    Write-LogLine ""
    Write-LogLine "=================================================="
    Write-LogLine $Name
    Write-LogLine "Started: $($started.ToString('o'))"
    Write-LogLine "Command: $CommandLine"
    Write-LogLine "=================================================="

    try {
        $global:LASTEXITCODE = 0
        & $Action 2>&1 | ForEach-Object {
            Write-LogLine ([string]$_)
        }
        if ($null -ne $script:StepExitCodeOverride) {
            $exitCode = [int]$script:StepExitCodeOverride
        }
        elseif ($UseLastExitCode -and $null -ne $LASTEXITCODE) {
            $exitCode = [int]$LASTEXITCODE
        }
    }
    catch {
        $exitCode = 1
        Write-LogLine "ERROR: $($_.Exception.Message)"
    }

    $finished = Get-Date
    Write-LogLine "Exit code: $exitCode"
    Write-LogLine "Finished: $($finished.ToString('o'))"

    $summary.Add([pscustomobject]@{
        Name = $Name
        ExitCode = $exitCode
        Started = $started.ToString("o")
        Finished = $finished.ToString("o")
    }) | Out-Null
}

function New-TempLogPath {
    param([string]$Prefix)

    $fileName = "{0}_{1}.txt" -f $Prefix, [System.Guid]::NewGuid().ToString("N")
    return Join-Path ([System.IO.Path]::GetTempPath()) $fileName
}

function Invoke-LoggedNative {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name,
        [Parameter(Mandatory = $true)]
        [string]$Command,
        [string[]]$Arguments = @()
    )

    $commandLine = Format-CommandLine -Command $Command -Arguments $Arguments
    Invoke-LoggedStep -Name $Name -CommandLine $commandLine -Action {
        $stdoutPath = New-TempLogPath -Prefix "frvp_stdout"
        $stderrPath = New-TempLogPath -Prefix "frvp_stderr"
        try {
            $process = Start-Process `
                -FilePath $Command `
                -ArgumentList $Arguments `
                -NoNewWindow `
                -Wait `
                -PassThru `
                -RedirectStandardOutput $stdoutPath `
                -RedirectStandardError $stderrPath

            if (Test-Path $stdoutPath) {
                Get-Content $stdoutPath
            }
            if (Test-Path $stderrPath) {
                Get-Content $stderrPath
            }
            $script:StepExitCodeOverride = [int]$process.ExitCode
        }
        finally {
            if (Test-Path $stdoutPath) {
                Remove-Item -LiteralPath $stdoutPath -Force
            }
            if (Test-Path $stderrPath) {
                Remove-Item -LiteralPath $stderrPath -Force
            }
        }
    }
}

function Invoke-LoggedScript {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name,
        [Parameter(Mandatory = $true)]
        [string]$ScriptPath,
        [hashtable]$Parameters = @{}
    )

    $parts = @($ScriptPath)
    foreach ($key in $Parameters.Keys) {
        $value = $Parameters[$key]
        if ($value -is [array]) {
            $parts += "-$key"
            $parts += ($value -join ",")
        }
        else {
            $parts += "-$key"
            $parts += [string]$value
        }
    }

    $commandLine = Format-CommandLine -Command "powershell-script" -Arguments $parts
    Invoke-LoggedStep -Name $Name -CommandLine $commandLine -UseLastExitCode -Action {
        & $ScriptPath @Parameters
    }
}

Push-Location $repoRoot
try {
    "" | Set-Content -Path $logPath -Encoding utf8
    Write-LogLine "FRVP setup-specific long rerun plan"
    Write-LogLine "Repo root: $repoRoot"
    Write-LogLine "Python: $PythonExe"
    Write-LogLine "Log file: $logPath"
    Write-LogLine "Run started: $((Get-Date).ToString('o'))"

    New-Item -ItemType Directory -Force "tmp\pytest" | Out-Null
    $pytestTempPath = (Resolve-Path "tmp\pytest").Path
    $env:TMP = $pytestTempPath
    $env:TEMP = $pytestTempPath
    $env:PYTEST_ADDOPTS = "-p no:cacheprovider"
    Write-LogLine "Pytest temp/cache path: $pytestTempPath"
    Write-LogLine "PYTEST_ADDOPTS: $env:PYTEST_ADDOPTS"

    Invoke-LoggedNative -Name "Step 1A: Verify per-setup target and setup detector tests" -Command $PythonExe -Arguments @(
        "-m", "pytest",
        "tests\test_frvp_per_setup_targets.py",
        "tests\test_frvp_setups.py"
    )

    Invoke-LoggedNative -Name "Step 1B: Verify registry and FRVP labeling tests" -Command $PythonExe -Arguments @(
        "-m", "pytest",
        "tests\test_build_frvp_candidate_registry.py",
        "tests\test_frvp_labeling.py"
    )

    Invoke-LoggedStep -Name "Step 4: Skip materialize setup-aware prepared root" -CommandLine "Prepared root already materialized at artifacts\frvp_es_primary_setup_targets_20260829\phase04\prepared" -Action {
        Write-Output "Skipping Phase 4 materialization; prepared root was already completed before this optimized rerun."
    }

    Invoke-LoggedStep -Name "Step 4B: List prepared target directories" -CommandLine "Get-ChildItem artifacts\frvp_es_primary_setup_targets_20260829\phase04\prepared -Directory | Select-Object Name" -Action {
        Get-ChildItem "artifacts\frvp_es_primary_setup_targets_20260829\phase04\prepared" -Directory | Select-Object Name
    }

    Invoke-LoggedNative -Name "Step 5A: Train full-span long continuation Setup 2" -Command $PythonExe -Arguments @(
        "-m", "model_training.ote_training.ote_xgboost_pipeline",
        "--prepared-root", "artifacts\frvp_es_primary_setup_targets_20260829\phase04\prepared",
        "--output-root", "models\frvp_long_continuation_setup2_xgb_20260829",
        "--backend", "xgboost",
        "--targets", "long_frvp_continuation_setup2",
        "--trials", "32",
        "--max-loaded-features", "160",
        "--top-feature-min", "24",
        "--top-feature-max", "88",
        "--window-min", "8",
        "--window-max", "32",
        "--event-tolerance-bars", "2",
        "--event-cooldown-bars", "4",
        "--calibration-method", "platt",
        "--cv-initial-train-rows", "400",
        "--cv-val-rows", "100",
        "--cv-step-rows", "100",
        "--cv-min-folds", "2",
        "--min-train-positive-rows", "70",
        "--min-val-positive-rows", "15",
        "--min-val-true-events", "8",
        "--threshold-event-fbeta-weight", "0.60",
        "--threshold-event-precision-weight", "0.40",
        "--threshold-turnover-penalty-weight", "0.35",
        "--threshold-turnover-target-ratio", "0.75",
        "--focal-alpha-min", "0.72",
        "--focal-alpha-max", "0.92",
        "--focal-gamma-min", "1.50",
        "--focal-gamma-max", "3.20",
        "--hard-negative-radius-min", "2",
        "--hard-negative-radius-max", "7",
        "--hard-negative-multiplier-min", "1.10",
        "--hard-negative-multiplier-max", "2.25",
        "--seed", "42"
    )

    Invoke-LoggedNative -Name "Step 5B: Train full-span long continuation Setup 3" -Command $PythonExe -Arguments @(
        "-m", "model_training.ote_training.ote_xgboost_pipeline",
        "--prepared-root", "artifacts\frvp_es_primary_setup_targets_20260829\phase04\prepared",
        "--output-root", "models\frvp_long_continuation_setup3_xgb_20260829",
        "--backend", "xgboost",
        "--targets", "long_frvp_continuation_setup3",
        "--trials", "36",
        "--max-loaded-features", "160",
        "--top-feature-min", "24",
        "--top-feature-max", "96",
        "--window-min", "8",
        "--window-max", "36",
        "--event-tolerance-bars", "2",
        "--event-cooldown-bars", "4",
        "--calibration-method", "platt",
        "--cv-initial-train-rows", "500",
        "--cv-val-rows", "120",
        "--cv-step-rows", "120",
        "--cv-min-folds", "2",
        "--min-train-positive-rows", "90",
        "--min-val-positive-rows", "18",
        "--min-val-true-events", "8",
        "--threshold-event-fbeta-weight", "0.60",
        "--threshold-event-precision-weight", "0.40",
        "--threshold-turnover-penalty-weight", "0.35",
        "--threshold-turnover-target-ratio", "0.75",
        "--focal-alpha-min", "0.72",
        "--focal-alpha-max", "0.92",
        "--focal-gamma-min", "1.50",
        "--focal-gamma-max", "3.20",
        "--hard-negative-radius-min", "2",
        "--hard-negative-radius-max", "7",
        "--hard-negative-multiplier-min", "1.10",
        "--hard-negative-multiplier-max", "2.25",
        "--seed", "42"
    )

    Invoke-LoggedNative -Name "Step 5C: Train full-span long continuation Setup 5" -Command $PythonExe -Arguments @(
        "-m", "model_training.ote_training.ote_xgboost_pipeline",
        "--prepared-root", "artifacts\frvp_es_primary_setup_targets_20260829\phase04\prepared",
        "--output-root", "models\frvp_long_continuation_setup5_xgb_20260829",
        "--backend", "xgboost",
        "--targets", "long_frvp_continuation_setup5",
        "--trials", "40",
        "--max-loaded-features", "160",
        "--top-feature-min", "24",
        "--top-feature-max", "112",
        "--window-min", "8",
        "--window-max", "40",
        "--event-tolerance-bars", "2",
        "--event-cooldown-bars", "4",
        "--calibration-method", "platt",
        "--cv-initial-train-rows", "650",
        "--cv-val-rows", "150",
        "--cv-step-rows", "150",
        "--cv-min-folds", "2",
        "--min-train-positive-rows", "120",
        "--min-val-positive-rows", "24",
        "--min-val-true-events", "10",
        "--threshold-event-fbeta-weight", "0.60",
        "--threshold-event-precision-weight", "0.40",
        "--threshold-turnover-penalty-weight", "0.35",
        "--threshold-turnover-target-ratio", "0.75",
        "--focal-alpha-min", "0.72",
        "--focal-alpha-max", "0.92",
        "--focal-gamma-min", "1.50",
        "--focal-gamma-max", "3.20",
        "--hard-negative-radius-min", "2",
        "--hard-negative-radius-max", "7",
        "--hard-negative-multiplier-min", "1.10",
        "--hard-negative-multiplier-max", "2.25",
        "--seed", "42"
    )

    Invoke-LoggedNative -Name "Step 6: Train full-span long reversal setups S1/S6" -Command $PythonExe -Arguments @(
        "-m", "model_training.ote_training.ote_xgboost_pipeline",
        "--prepared-root", "artifacts\frvp_es_primary_setup_targets_20260829\phase04\prepared",
        "--output-root", "models\frvp_long_reversal_setup_fullspan_rerun2_xgb_20260829",
        "--backend", "xgboost",
        "--targets", "long_frvp_reversal_setup1", "long_frvp_reversal_setup6",
        "--trials", "32",
        "--max-loaded-features", "160",
        "--top-feature-min", "24",
        "--top-feature-max", "96",
        "--window-min", "8",
        "--window-max", "32",
        "--event-tolerance-bars", "2",
        "--event-cooldown-bars", "4",
        "--calibration-method", "platt",
        "--cv-initial-train-rows", "350",
        "--cv-val-rows", "100",
        "--cv-step-rows", "100",
        "--cv-min-folds", "2",
        "--min-train-positive-rows", "70",
        "--min-val-positive-rows", "15",
        "--min-val-true-events", "8",
        "--seed", "42"
    )

    Invoke-LoggedNative -Name "Step 7: Train Setup 4 standalone" -Command $PythonExe -Arguments @(
        "-m", "model_training.ote_training.ote_xgboost_pipeline",
        "--prepared-root", "artifacts\frvp_es_primary_setup_targets_20260829\phase04\prepared",
        "--output-root", "models\frvp_long_reversal_setup4_xgb_20260829",
        "--backend", "xgboost",
        "--targets", "long_frvp_reversal_setup4",
        "--trials", "40",
        "--max-loaded-features", "160",
        "--top-feature-min", "12",
        "--top-feature-max", "48",
        "--window-min", "8",
        "--window-max", "16",
        "--event-tolerance-bars", "2",
        "--event-cooldown-bars", "4",
        "--calibration-method", "platt",
        "--cv-initial-train-rows", "105",
        "--cv-val-rows", "35",
        "--cv-step-rows", "35",
        "--cv-min-folds", "1",
        "--min-train-positive-rows", "18",
        "--min-val-positive-rows", "4",
        "--min-val-true-events", "3",
        "--final-eval-min-rows", "40",
        "--seed", "42"
    )

    if (Test-Path "artifacts\frvp_long_reversal_setup1_recency730_20260829\phase04\prepared") {
        Invoke-LoggedStep -Name "Step 8A: Skip recency-weighted reversal Setup 1 root" -CommandLine "Prepared root already materialized at artifacts\frvp_long_reversal_setup1_recency730_20260829\phase04\prepared" -Action {
            Write-Output "Skipping S1 recency prepared root; already exists."
        }
    }
    else {
        Invoke-LoggedNative -Name "Step 8A: Materialize recency-weighted reversal Setup 1 root" -Command $PythonExe -Arguments @(
            "scripts\materialize_frvp_recency_prepared_root.py",
            "--base-prepared-root", "artifacts\frvp_es_primary_setup_targets_20260829\phase04\prepared",
            "--output-prepared-root", "artifacts\frvp_long_reversal_setup1_recency730_20260829\phase04\prepared",
            "--target", "long_frvp_reversal_setup1",
            "--half-life-days", "730",
            "--floor", "0.20"
        )
    }

    if (Test-Path "artifacts\frvp_long_reversal_setup6_recency730_20260829\phase04\prepared") {
        Invoke-LoggedStep -Name "Step 8B: Skip recency-weighted reversal Setup 6 root" -CommandLine "Prepared root already materialized at artifacts\frvp_long_reversal_setup6_recency730_20260829\phase04\prepared" -Action {
            Write-Output "Skipping S6 recency prepared root; already exists."
        }
    }
    else {
        Invoke-LoggedNative -Name "Step 8B: Materialize recency-weighted reversal Setup 6 root" -Command $PythonExe -Arguments @(
            "scripts\materialize_frvp_recency_prepared_root.py",
            "--base-prepared-root", "artifacts\frvp_es_primary_setup_targets_20260829\phase04\prepared",
            "--output-prepared-root", "artifacts\frvp_long_reversal_setup6_recency730_20260829\phase04\prepared",
            "--target", "long_frvp_reversal_setup6",
            "--half-life-days", "730",
            "--floor", "0.20"
        )
    }

    Invoke-LoggedNative -Name "Step 9A: Train recency-weighted reversal Setup 1 optimized challenger" -Command $PythonExe -Arguments @(
        "-m", "model_training.ote_training.ote_xgboost_pipeline",
        "--prepared-root", "artifacts\frvp_long_reversal_setup1_recency730_20260829\phase04\prepared",
        "--output-root", "models\frvp_long_reversal_setup1_recency730_opt_xgb_20260829",
        "--backend", "xgboost",
        "--targets", "long_frvp_reversal_setup1",
        "--trials", "56",
        "--max-loaded-features", "160",
        "--top-feature-min", "32",
        "--top-feature-max", "112",
        "--window-min", "8",
        "--window-max", "20",
        "--event-tolerance-bars", "2",
        "--event-cooldown-bars", "4",
        "--calibration-method", "platt",
        "--cv-initial-train-rows", "350",
        "--cv-val-rows", "100",
        "--cv-step-rows", "100",
        "--cv-min-folds", "2",
        "--min-train-positive-rows", "70",
        "--min-val-positive-rows", "15",
        "--min-val-true-events", "8",
        "--threshold-event-fbeta-weight", "0.35",
        "--threshold-event-precision-weight", "0.65",
        "--threshold-turnover-penalty-weight", "0.75",
        "--threshold-turnover-target-ratio", "0.45",
        "--objective-average-precision-weight", "0.35",
        "--objective-threshold-score-weight", "0.55",
        "--objective-brier-penalty-weight", "0.10",
        "--focal-alpha-min", "0.76",
        "--focal-alpha-max", "0.92",
        "--focal-gamma-min", "1.70",
        "--focal-gamma-max", "2.80",
        "--hard-negative-radius-min", "4",
        "--hard-negative-radius-max", "8",
        "--hard-negative-multiplier-min", "1.50",
        "--hard-negative-multiplier-max", "2.50",
        "--seed", "42"
    )

    Invoke-LoggedNative -Name "Step 9B: Train recency-weighted reversal Setup 6 recall challenger" -Command $PythonExe -Arguments @(
        "-m", "model_training.ote_training.ote_xgboost_pipeline",
        "--prepared-root", "artifacts\frvp_long_reversal_setup6_recency730_20260829\phase04\prepared",
        "--output-root", "models\frvp_long_reversal_setup6_recency730_recall_xgb_20260829",
        "--backend", "xgboost",
        "--targets", "long_frvp_reversal_setup6",
        "--trials", "56",
        "--max-loaded-features", "160",
        "--top-feature-min", "24",
        "--top-feature-max", "128",
        "--window-min", "12",
        "--window-max", "40",
        "--event-tolerance-bars", "2",
        "--event-cooldown-bars", "4",
        "--calibration-method", "platt",
        "--cv-initial-train-rows", "350",
        "--cv-val-rows", "100",
        "--cv-step-rows", "100",
        "--cv-min-folds", "2",
        "--min-train-positive-rows", "70",
        "--min-val-positive-rows", "15",
        "--min-val-true-events", "8",
        "--threshold-event-fbeta-weight", "0.75",
        "--threshold-event-precision-weight", "0.25",
        "--threshold-turnover-penalty-weight", "0.20",
        "--threshold-turnover-target-ratio", "0.95",
        "--objective-average-precision-weight", "0.50",
        "--objective-threshold-score-weight", "0.45",
        "--objective-brier-penalty-weight", "0.05",
        "--focal-alpha-min", "0.82",
        "--focal-alpha-max", "0.96",
        "--focal-gamma-min", "1.25",
        "--focal-gamma-max", "2.40",
        "--hard-negative-radius-min", "1",
        "--hard-negative-radius-max", "4",
        "--hard-negative-multiplier-min", "1.00",
        "--hard-negative-multiplier-max", "1.60",
        "--seed", "42"
    )

    Invoke-LoggedNative -Name "Step 10A: Build full-span setup registry" -Command $PythonExe -Arguments @(
        "scripts\build_frvp_candidate_registry.py",
        "--model-root", "models\frvp_long_continuation_setup2_xgb_20260829",
        "--model-root", "models\frvp_long_continuation_setup3_xgb_20260829",
        "--model-root", "models\frvp_long_continuation_setup5_xgb_20260829",
        "--model-root", "models\frvp_long_reversal_setup_fullspan_rerun2_xgb_20260829",
        "--model-root", "models\frvp_long_reversal_setup4_xgb_20260829",
        "--source-registry-path", "models\frvp_es_primary_model_registry_current.json",
        "--output-path", "models\frvp_es_primary_model_registry_long_setup_fullspan_20260829.json"
    )

    Invoke-LoggedNative -Name "Step 10B: Build optimized recency reversal setup registry" -Command $PythonExe -Arguments @(
        "scripts\build_frvp_candidate_registry.py",
        "--model-root", "models\frvp_long_reversal_setup1_recency730_opt_xgb_20260829",
        "--model-root", "models\frvp_long_reversal_setup6_recency730_recall_xgb_20260829",
        "--source-registry-path", "models\frvp_es_primary_model_registry_current.json",
        "--output-path", "models\frvp_es_primary_model_registry_long_reversal_setup_recency730_opt_20260829.json"
    )

    Invoke-LoggedScript -Name "Step 11: Evaluate full-span regular lanes" -ScriptPath ".\scripts\run_frvp_post_training_eval.ps1" -Parameters @{
        RegistryPath = "models\frvp_es_primary_model_registry_long_setup_fullspan_20260829.json"
        RegimeOutputRoot = "model_testing\reports\frvp_regime_slices\frvp_long_setup_fullspan_20260829"
        ThresholdOutputRoot = "model_testing\reports\frvp_threshold_policies\frvp_long_setup_fullspan_20260829"
        BacktestOutputRoot = "model_testing\reports\frvp_backtests\frvp_long_setup_fullspan_20260829"
        ModelIds = @(
            "frvp_long_reversal_setup1_xgb_v1",
            "frvp_long_reversal_setup6_xgb_v1",
            "frvp_long_continuation_setup2_xgb_v1",
            "frvp_long_continuation_setup3_xgb_v1",
            "frvp_long_continuation_setup5_xgb_v1"
        )
        BacktestMinTrainYears = 2
        BacktestMinFolds = 8
        SpreadCostMode = "session_schedule"
    }

    Invoke-LoggedNative -Name "Step 12A: Setup 4 regime-slice report" -Command $PythonExe -Arguments @(
        "scripts\run_ote_regime_slice_report.py",
        "--registry-path", "models\frvp_es_primary_model_registry_long_setup_fullspan_20260829.json",
        "--output-root", "model_testing\reports\frvp_regime_slices\frvp_long_setup4_fullspan_20260829",
        "--status", "candidate",
        "--model-id", "frvp_long_reversal_setup4_xgb_v1",
        "--bootstrap-iterations", "200",
        "--min-positive-events", "10"
    )

    Invoke-LoggedNative -Name "Step 12B: Setup 4 threshold policy search" -Command $PythonExe -Arguments @(
        "scripts\run_ote_threshold_policy_search.py",
        "--regime-report-root", "model_testing\reports\frvp_regime_slices\frvp_long_setup4_fullspan_20260829",
        "--registry-path", "models\frvp_es_primary_model_registry_long_setup_fullspan_20260829.json",
        "--output-root", "model_testing\reports\frvp_threshold_policies\frvp_long_setup4_fullspan_20260829",
        "--status", "candidate",
        "--model-id", "frvp_long_reversal_setup4_xgb_v1",
        "--instrument", "es",
        "--spread-cost-mode", "session_schedule",
        "--min-positive-events", "10",
        "--min-events-per-month", "0.5",
        "--min-trades-per-week", "0.5",
        "--evaluation-contract-mode", "research",
        "--write-policy-decisions"
    )

    Invoke-LoggedNative -Name "Step 12C: Setup 4 policy backtest" -Command $PythonExe -Arguments @(
        "scripts\run_ote_policy_backtest.py",
        "--regime-report-root", "model_testing\reports\frvp_regime_slices\frvp_long_setup4_fullspan_20260829",
        "--registry-path", "models\frvp_es_primary_model_registry_long_setup_fullspan_20260829.json",
        "--output-root", "model_testing\reports\frvp_backtests\frvp_long_setup4_fullspan_20260829",
        "--status", "candidate",
        "--model-id", "frvp_long_reversal_setup4_xgb_v1",
        "--instrument", "es",
        "--spread-cost-mode", "session_schedule",
        "--min-train-years", "2",
        "--test-window-months", "3",
        "--rolling-step-months", "3",
        "--min-folds", "4",
        "--min-positive-events", "10",
        "--min-events-per-month", "0.5",
        "--min-trades-per-week", "0.5",
        "--evaluation-contract-mode", "research",
        "--minimum-sharpe", "0.8",
        "--maximum-drawdown-pct", "12.0",
        "--minimum-dsr", "0.3"
    )

    Invoke-LoggedScript -Name "Step 13A: Evaluate optimized recency reversal lanes full window" -ScriptPath ".\scripts\run_frvp_post_training_eval.ps1" -Parameters @{
        RegistryPath = "models\frvp_es_primary_model_registry_long_reversal_setup_recency730_opt_20260829.json"
        RegimeOutputRoot = "model_testing\reports\frvp_regime_slices\frvp_long_reversal_setup_recency730_opt_20260829"
        ThresholdOutputRoot = "model_testing\reports\frvp_threshold_policies\frvp_long_reversal_setup_recency730_opt_20260829"
        BacktestOutputRoot = "model_testing\reports\frvp_backtests\frvp_long_reversal_setup_recency730_opt_20260829"
        ModelIds = @(
            "frvp_long_reversal_setup1_xgb_v1",
            "frvp_long_reversal_setup6_xgb_v1"
        )
        BacktestMinTrainYears = 2
        BacktestMinFolds = 8
        SpreadCostMode = "session_schedule"
    }

    Invoke-LoggedScript -Name "Step 13B: Evaluate optimized recency reversal lanes recent2y" -ScriptPath ".\scripts\run_frvp_post_training_eval.ps1" -Parameters @{
        RegistryPath = "models\frvp_es_primary_model_registry_long_reversal_setup_recency730_opt_20260829.json"
        RegimeOutputRoot = "model_testing\reports\frvp_regime_slices\frvp_long_reversal_setup_recency730_opt_recent2y_20260829"
        ThresholdOutputRoot = "model_testing\reports\frvp_threshold_policies\frvp_long_reversal_setup_recency730_opt_recent2y_20260829"
        BacktestOutputRoot = "model_testing\reports\frvp_backtests\frvp_long_reversal_setup_recency730_opt_recent2y_20260829"
        ModelIds = @(
            "frvp_long_reversal_setup1_xgb_v1",
            "frvp_long_reversal_setup6_xgb_v1"
        )
        BacktestMinTrainYears = 2
        BacktestMaxTrainYears = 2
        BacktestMinScheduledTestStart = "2024-01-01"
        BacktestMinFolds = 8
        SpreadCostMode = "session_schedule"
    }
}
finally {
    Write-LogLine ""
    Write-LogLine "=================================================="
    Write-LogLine "Run Summary"
    Write-LogLine "=================================================="
    foreach ($item in $summary) {
        Write-LogLine ("{0} | exit={1} | started={2} | finished={3}" -f $item.Name, $item.ExitCode, $item.Started, $item.Finished)
    }
    Write-LogLine "Run finished: $((Get-Date).ToString('o'))"
    Pop-Location
}
