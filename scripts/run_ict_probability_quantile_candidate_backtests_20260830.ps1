param(
    [switch]$SkipExisting
)

$ErrorActionPreference = "Stop"

$PythonExe = "ote_venv\Scripts\python.exe"
$BacktestScript = "scripts\run_ote_policy_backtest.py"
$RegistryPath = "models\ict_es_primary_model_registry_bootstrap_20260726_full.json"
$RegimeRoot = "model_testing\reports\ict_regime_slices\ict_es_primary_bootstrap_20260726_full"
$OutputBase = "model_testing\reports\ict_backtests\ict_probability_quantile_candidate_sweep_20260830"

$Runs = @(
    @{
        Label = "ict_long_continuation_q40"
        ModelId = "ict_long_continuation_xgb_v1"
        Preset = "ict_long_continuation_xgb_probability_quantile_q40_v1"
    },
    @{
        Label = "ict_long_continuation_q50"
        ModelId = "ict_long_continuation_xgb_v1"
        Preset = "ict_long_continuation_xgb_probability_quantile_q50_v1"
    },
    @{
        Label = "ict_long_continuation_q60"
        ModelId = "ict_long_continuation_xgb_v1"
        Preset = "ict_long_continuation_xgb_probability_quantile_q60_v1"
    },
    @{
        Label = "ict_short_continuation_q50"
        ModelId = "ict_short_continuation_xgb_v1"
        Preset = "ict_short_continuation_xgb_probability_quantile_q50_v1"
    },
    @{
        Label = "ict_short_continuation_q60"
        ModelId = "ict_short_continuation_xgb_v1"
        Preset = "ict_short_continuation_xgb_probability_quantile_q60_v1"
    }
)

New-Item -ItemType Directory -Force $OutputBase | Out-Null

foreach ($Run in $Runs) {
    $OutputRoot = Join-Path $OutputBase $Run.Label
    $SummaryPath = Join-Path $OutputRoot "model_summary.csv"

    if ($SkipExisting -and (Test-Path $SummaryPath)) {
        Write-Output "Skipping $($Run.Label); model_summary.csv already exists."
        continue
    }

    Write-Output "Running $($Run.Label): $($Run.ModelId) with $($Run.Preset)"

    $Arguments = @(
        $BacktestScript,
        "--regime-report-root", $RegimeRoot,
        "--registry-path", $RegistryPath,
        "--output-root", $OutputRoot,
        "--status", "candidate",
        "--model-id", $Run.ModelId,
        "--min-train-years", "2",
        "--min-folds", "8",
        "--min-positive-events", "50",
        "--min-events-per-month", "3.0",
        "--min-trades-per-week", "3.0",
        "--instrument", "es",
        "--spread-cost-mode", "session_schedule",
        "--targeted-filter-preset", $Run.Preset
    )

    & $PythonExe @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Backtest failed for $($Run.Label) with exit code $LASTEXITCODE."
    }
}

$AggregateRows = @()
foreach ($Run in $Runs) {
    $OutputRoot = Join-Path $OutputBase $Run.Label
    $SummaryPath = Join-Path $OutputRoot "model_summary.csv"
    if (-not (Test-Path $SummaryPath)) {
        continue
    }

    foreach ($Row in (Import-Csv $SummaryPath)) {
        $Row | Add-Member -NotePropertyName "sweep_label" -NotePropertyValue $Run.Label -Force
        $Row | Add-Member -NotePropertyName "targeted_filter_preset" -NotePropertyValue $Run.Preset -Force
        $Row | Add-Member -NotePropertyName "sweep_output_root" -NotePropertyValue $OutputRoot -Force
        $AggregateRows += $Row
    }
}

if ($AggregateRows.Count -gt 0) {
    $AggregatePath = Join-Path $OutputBase "candidate_quantile_sweep_summary.csv"
    $AggregateRows | Export-Csv $AggregatePath -NoTypeInformation
    Write-Output "Wrote aggregate summary: $AggregatePath"
}

Write-Output "Done."
