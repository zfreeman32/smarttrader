param(
    [string]$PythonExe = "",
    [string]$InputPath = "data/currency_data/eurusd-1m.csv",
    [string]$NormalizedInputPath = "data/currency_data/eurusd-1m.normalized.csv",
    [string]$LabelsOutput = "data/labeling/labeled_data/eurusd_1min_ote_labels_full.csv",
    [string]$SwingsOutput = "data/labeling/labeled_data/eurusd_1min_ote_swings_full.csv",
    [string]$FeaturesOutput = "data/features/eurusd_1min_ote_full.csv",
    [string]$PreparedRoot = "data/prepared/eurusd_1min_ote_full",
    [string]$Recipe = "features/recipes/ote_extended.json",
    [string]$SourceTimezone = "GMT-6",
    [string]$CanonicalTimezone = "UTC",
    [string]$FeatureClockTimezone = "America/New_York",
    [string]$MarketCloseTimezone = "America/New_York",
    [string]$BarTimestampSemantics = "as_provided",
    [string]$StartDate = "2019-01-01",
    [string]$EndDate = "",
    [string]$Scaler = "none",
    [double]$CorrThreshold = 0.98,
    [double]$SimilarityThreshold = 0.995,
    [int]$MaxAnalysisRows = 100000,
    [int]$TransformWorkers = 4,
    [int]$StrategyTimeoutSeconds = 600,
    [switch]$UseAllStrategies,
    [switch]$StrictStrategyErrors,
    [switch]$SkipNormalization,
    [switch]$ForceNormalization,
    [switch]$NormalizeOnly,
    [switch]$SkipLabeling,
    [switch]$SkipFeatures,
    [switch]$SkipPreprocessing,
    [switch]$Plot,
    [string[]]$Strategies = @(),
    [string[]]$Targets = @()
)

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

function Resolve-RepoPath {
    param([string]$RelativePath)

    if ([System.IO.Path]::IsPathRooted($RelativePath)) {
        return $RelativePath
    }

    return Join-Path $repoRoot $RelativePath
}

function Invoke-Step {
    param(
        [string]$Name,
        [string[]]$Command
    )

    Write-Host ""
    Write-Host $Name
    Write-Host ("Command: " + ($Command -join " "))

    & $Command[0] $Command[1..($Command.Count - 1)]
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE."
    }
}

$resolvedInputPath = Resolve-RepoPath $InputPath
$resolvedNormalizedInputPath = Resolve-RepoPath $NormalizedInputPath
$resolvedLabelsOutput = Resolve-RepoPath $LabelsOutput
$resolvedSwingsOutput = Resolve-RepoPath $SwingsOutput
$resolvedFeaturesOutput = Resolve-RepoPath $FeaturesOutput
$resolvedPreparedRoot = Resolve-RepoPath $PreparedRoot
$resolvedRecipe = Resolve-RepoPath $Recipe

if (-not (Test-Path $resolvedInputPath)) {
    throw "Input file does not exist: $resolvedInputPath"
}

New-Item -ItemType Directory -Force -Path (Split-Path $resolvedNormalizedInputPath -Parent) | Out-Null
New-Item -ItemType Directory -Force -Path (Split-Path $resolvedLabelsOutput -Parent) | Out-Null
New-Item -ItemType Directory -Force -Path (Split-Path $resolvedSwingsOutput -Parent) | Out-Null
New-Item -ItemType Directory -Force -Path (Split-Path $resolvedFeaturesOutput -Parent) | Out-Null
New-Item -ItemType Directory -Force -Path $resolvedPreparedRoot | Out-Null

Write-Host "=================================================="
Write-Host "EURUSD 1-Minute Label -> Features -> Preprocessing"
Write-Host "Working directory: $repoRoot"
Write-Host "Python: $PythonExe"
Write-Host "Raw input: $resolvedInputPath"
Write-Host "Normalized input: $resolvedNormalizedInputPath"
Write-Host "Labels output: $resolvedLabelsOutput"
Write-Host "Features output: $resolvedFeaturesOutput"
Write-Host "Prepared root: $resolvedPreparedRoot"
Write-Host "Recipe: $resolvedRecipe"
Write-Host "Raw source timezone: $SourceTimezone"
Write-Host "Canonical timezone: $CanonicalTimezone"
Write-Host "Feature clock timezone: $FeatureClockTimezone"
Write-Host "Market close timezone: $MarketCloseTimezone"
Write-Host "Start date: $StartDate"
if ([string]::IsNullOrWhiteSpace($EndDate)) {
    Write-Host "End date: latest bar in source"
}
else {
    Write-Host "End date: $EndDate"
}
Write-Host "=================================================="

$labelingInputPath = $resolvedInputPath

Push-Location $repoRoot
try {
    if (-not $SkipNormalization) {
        $reuseNormalized = (
            -not $ForceNormalization `
            -and (Test-Path $resolvedNormalizedInputPath) `
            -and ((Get-Item $resolvedNormalizedInputPath).LastWriteTimeUtc -ge (Get-Item $resolvedInputPath).LastWriteTimeUtc)
        )

        if ($reuseNormalized) {
            Write-Host ""
            Write-Host "[0/3] Reusing existing normalized 1-minute CSV."
        }
        else {
            $normalizeCommand = @(
                $PythonExe,
                "scripts/normalize_fx_csv.py",
                $resolvedInputPath,
                $resolvedNormalizedInputPath,
                "--overwrite"
            )
            Invoke-Step "[0/3] Normalizing raw 1-minute CSV..." $normalizeCommand
        }

        $labelingInputPath = $resolvedNormalizedInputPath
    }

    if ($NormalizeOnly) {
        Write-Host ""
        Write-Host "Normalization completed. Exiting because -NormalizeOnly was supplied."
        exit 0
    }

    if (-not $SkipLabeling) {
        $labelCommand = @(
            $PythonExe,
            "data/labeling/labeling_engine.py",
            "--input", $labelingInputPath,
            "--output", $resolvedLabelsOutput,
            "--swings-output", $resolvedSwingsOutput,
            "--source-timezone", $SourceTimezone,
            "--canonical-timezone", $CanonicalTimezone,
            "--bar-timestamp-semantics", $BarTimestampSemantics
        )

        if (-not [string]::IsNullOrWhiteSpace($StartDate)) {
            $labelCommand += @("--start-date", $StartDate)
        }
        if (-not [string]::IsNullOrWhiteSpace($EndDate)) {
            $labelCommand += @("--end-date", $EndDate)
        }
        if ($Plot) {
            $labelCommand += "--plot"
        }

        Invoke-Step "[1/3] Running labeling engine..." $labelCommand
    }
    elseif (-not (Test-Path $resolvedLabelsOutput)) {
        throw "Skipping labeling was requested, but the labels file does not exist: $resolvedLabelsOutput"
    }

    if (-not $SkipFeatures) {
        $featureCommand = @(
            $PythonExe,
            "-m",
            "features.cli",
            "build",
            $resolvedLabelsOutput,
            "--output", $resolvedFeaturesOutput,
            "--recipe", $resolvedRecipe,
            "--transform-workers", $TransformWorkers,
            "--optimize-memory",
            "--progress",
            "--strategy-timeout-seconds", $StrategyTimeoutSeconds,
            "--source-timezone", $CanonicalTimezone,
            "--canonical-timezone", $CanonicalTimezone,
            "--feature-clock-timezone", $FeatureClockTimezone,
            "--market-close-timezone", $MarketCloseTimezone
        )

        if ($UseAllStrategies) {
            $featureCommand += "--all-strategies"
        }
        foreach ($strategy in $Strategies) {
            $featureCommand += @("--strategy", $strategy)
        }
        if (-not $StrictStrategyErrors) {
            $featureCommand += "--skip-strategy-errors"
        }

        Invoke-Step "[2/3] Building features..." $featureCommand
    }
    elseif (-not (Test-Path $resolvedFeaturesOutput)) {
        throw "Skipping feature generation was requested, but the feature file does not exist: $resolvedFeaturesOutput"
    }

    if (-not $SkipPreprocessing) {
        $prepareCommand = @(
            $PythonExe,
            "-m",
            "preprocessing",
            "prepare",
            $resolvedFeaturesOutput,
            "--output-dir", $resolvedPreparedRoot,
            "--scaler", $Scaler,
            "--corr-threshold", $CorrThreshold,
            "--similarity-threshold", $SimilarityThreshold,
            "--max-analysis-rows", $MaxAnalysisRows
        )

        foreach ($target in $Targets) {
            $prepareCommand += @("--target", $target)
        }

        Invoke-Step "[3/3] Preparing training-ready datasets..." $prepareCommand
    }
    else {
        Write-Host ""
        Write-Host "Preprocessing skipped."
    }

    Write-Host ""
    Write-Host "1-minute pipeline completed successfully."
}
finally {
    Pop-Location
}
