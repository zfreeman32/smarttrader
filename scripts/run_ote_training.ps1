param(
    [string]$PythonExe = "",
    [string]$PreparedRoot = "",
    [string]$OutputRoot = "",
    [string]$Backend = "",
    [string]$ModelType = "",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Targets
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

if ([string]::IsNullOrWhiteSpace($PreparedRoot)) {
    $PreparedRoot = "data/prepared/eurusd_5min_ote_full"
}
if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $OutputRoot = "models/ote_multi_family_tcn_v1"
}
if ([string]::IsNullOrWhiteSpace($Backend)) {
    $Backend = "torch"
}
if ([string]::IsNullOrWhiteSpace($ModelType)) {
    $ModelType = "tcn"
}

Write-Host "=================================================="
Write-Host "OTE Multi-Family Training"
Write-Host "Working directory: $repoRoot"
Write-Host "Python: $PythonExe"
Write-Host "Prepared root: $PreparedRoot"
Write-Host "Output root: $OutputRoot"
Write-Host "Backend: $Backend"
Write-Host "Model type: $ModelType"
Write-Host "=================================================="
Write-Host ""

$command = @(
    "-m",
    "model_training.ote_training.ote_xgboost_pipeline",
    "--prepared-root", $PreparedRoot,
    "--output-root", $OutputRoot,
    "--backend", $Backend,
    "--model-type", $ModelType
)

if ($Targets -and $Targets.Count -gt 0) {
    $helpFlags = @("-h", "--help")
    if ($Targets.Count -eq 1 -and $helpFlags -contains $Targets[0]) {
        $command += $Targets[0]
    }
    else {
        Write-Host ("Training requested target(s): " + ($Targets -join " "))
        $command += "--targets"
        $command += $Targets
    }
}
else {
    Write-Host "No explicit targets supplied. Auto-discovering all prepared target directories..."
}

Push-Location $repoRoot
try {
    & $PythonExe @command
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
