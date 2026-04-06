param(
    [string]$PythonExe = "",
    [string]$ServiceName = "ote-live-signal-service",
    [string]$DbPath = "",
    [string]$LogFile = "",
    [string]$HeartbeatFile = "",
    [double]$PollIntervalSeconds = 60.0,
    [double]$MinDiskFreeGb = 2.0,
    [int]$RestartDelaySeconds = 15,
    [int]$MaxRestartCount = 10,
    [int]$MaxRestartsPerHour = 6,
    [switch]$NoRestartOnFailure,
    [switch]$DisableSignalRuntime,
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

if ([string]::IsNullOrWhiteSpace($DbPath)) {
    $DbPath = Join-Path $repoRoot "ote_live\runtime_data\live_market_data.sqlite3"
}
if ([string]::IsNullOrWhiteSpace($LogFile)) {
    $LogFile = Join-Path $repoRoot "ote_live\runtime_data\logs\live_signal_service.log"
}
if ([string]::IsNullOrWhiteSpace($HeartbeatFile)) {
    $HeartbeatFile = Join-Path $repoRoot "ote_live\runtime_data\health\live_signal_service_heartbeat.json"
}

$command = @(
    "-m",
    "ote_live.scripts.run_live_collector",
    "--service-name", $ServiceName,
    "--db-path", $DbPath,
    "--log-file", $LogFile,
    "--heartbeat-file", $HeartbeatFile,
    "--poll-interval-seconds", $PollIntervalSeconds,
    "--min-disk-free-gb", $MinDiskFreeGb
)

if ($DisableSignalRuntime) {
    $command += "--disable-signal-runtime"
}
if ($ExtraArgs) {
    $command += $ExtraArgs
}

Push-Location $repoRoot
try {
    $restartOnFailure = -not $NoRestartOnFailure.IsPresent
    $restartCount = 0
    $restartTimestamps = @()

    while ($true) {
        Write-Host "[$(Get-Date -Format o)] Starting live signal service '$ServiceName'"
        & $PythonExe @command
        $exitCode = $LASTEXITCODE

        if ($exitCode -eq 0) {
            exit 0
        }
        if (-not $restartOnFailure) {
            exit $exitCode
        }

        $now = Get-Date
        $restartTimestamps = @($restartTimestamps | Where-Object { $_ -gt $now.AddHours(-1) })
        $restartTimestamps += $now

        if ($restartCount -ge $MaxRestartCount) {
            Write-Error "Live signal service exceeded MaxRestartCount=$MaxRestartCount. Last exit code: $exitCode"
            exit $exitCode
        }
        if ($restartTimestamps.Count -gt $MaxRestartsPerHour) {
            Write-Error "Live signal service exceeded MaxRestartsPerHour=$MaxRestartsPerHour. Last exit code: $exitCode"
            exit $exitCode
        }

        $restartCount += 1
        Write-Warning "Live signal service exited with code $exitCode. Restart attempt $restartCount of $MaxRestartCount in $RestartDelaySeconds seconds."
        Start-Sleep -Seconds $RestartDelaySeconds
    }
}
finally {
    Pop-Location
}
