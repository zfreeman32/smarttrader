param(
    [string]$PythonExe = "",
    [switch]$PackagePolicies,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
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

Push-Location $repoRoot
try {
    $launchArgs = @()
    if (-not $PackagePolicies) {
        $launchArgs += "--skip-policy-package"
    }
    $launchArgs += $ExtraArgs

    & $PythonExe -m ote_live.scripts.run_live_stack @launchArgs
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
