param(
    [string]$NgrokUrl = "https://stencil-exporter-pound.ngrok-free.dev",
    [switch]$IncludeIct,
    [switch]$PrepareIctHandoff
)

$ErrorActionPreference = "Stop"

$repoRoot = $PSScriptRoot
$pythonExe = Join-Path $repoRoot "ote_venv\Scripts\python.exe"
$logRoot = Join-Path $repoRoot "ote_live\runtime_data\logs"
$healthRoot = Join-Path $repoRoot "ote_live\runtime_data\health"

if (-not (Test-Path -LiteralPath $pythonExe)) {
    throw "Python virtual environment was not found at $pythonExe"
}

New-Item -ItemType Directory -Force -Path $logRoot | Out-Null

function Test-HeartbeatProcess {
    param([Parameter(Mandatory)][string]$HeartbeatPath)

    if (-not (Test-Path -LiteralPath $HeartbeatPath)) {
        return $false
    }

    try {
        $heartbeat = Get-Content -Raw -LiteralPath $HeartbeatPath | ConvertFrom-Json
        $processId = [int]$heartbeat.process_id
        $process = Get-Process -Id $processId -ErrorAction Stop
        return $process.ProcessName -eq "python"
    }
    catch {
        return $false
    }
}

function Test-CleanIctHandoffSnapshot {
    param([Parameter(Mandatory)][string]$HeartbeatPath)

    if (-not (Test-Path -LiteralPath $HeartbeatPath)) {
        return $false
    }

    try {
        $heartbeat = Get-Content -Raw -LiteralPath $HeartbeatPath | ConvertFrom-Json
        if ($heartbeat.service_name -ne "es-shared-live-signal-service" -or
            $heartbeat.service_status -ne "stopped" -or
            $heartbeat.health_state -ne "healthy" -or
            $heartbeat.heartbeat_is_stale -ne $false -or
            $heartbeat.asset -ne "ES" -or
            $heartbeat.source_timeframe -ne "5m" -or
            $heartbeat.signal_timeframe -ne "5m") {
            return $false
        }

        $generatedAt = [DateTimeOffset]::Parse([string]$heartbeat.generated_at_utc)
        $ageSeconds = ([DateTimeOffset]::UtcNow - $generatedAt.ToUniversalTime()).TotalSeconds
        if ($ageSeconds -lt -60 -or $ageSeconds -gt 120) {
            return $false
        }

        $recordedProcessId = [int]$heartbeat.process_id
        if ($recordedProcessId -le 0) {
            return $false
        }
        $recordedProcess = Get-Process -Id $recordedProcessId -ErrorAction SilentlyContinue
        return $null -eq $recordedProcess
    }
    catch {
        return $false
    }
}

function Test-LocalPort {
    param([Parameter(Mandatory)][int]$Port)

    $client = [System.Net.Sockets.TcpClient]::new()
    try {
        $connection = $client.ConnectAsync("127.0.0.1", $Port)
        return $connection.Wait(1000) -and $client.Connected
    }
    catch {
        return $false
    }
    finally {
        $client.Dispose()
    }
}

function Stop-AppModuleProcesses {
    param(
        [Parameter(Mandatory)][string]$Name,
        [Parameter(Mandatory)][string]$Module
    )

    $processes = @(Get-CimInstance Win32_Process -ErrorAction Stop | Where-Object {
        $_.Name -match '^python(w)?\.exe$' -and
        -not [string]::IsNullOrWhiteSpace($_.CommandLine) -and
        $_.CommandLine -like "*${Module}*"
    })

    foreach ($processInfo in $processes) {
        $processId = [int]$processInfo.ProcessId
        $process = Get-Process -Id $processId -ErrorAction SilentlyContinue
        if ($null -eq $process) {
            continue
        }

        Write-Host "Stopping $Name (PID $processId)..."
        Stop-Process -Id $processId -ErrorAction Stop
        $process.WaitForExit(10000) | Out-Null
        if (-not $process.HasExited) {
            throw "$Name (PID $processId) did not stop within 10 seconds."
        }
    }

    return $processes.Count
}

function Stop-DashboardNgrokProcesses {
    $processes = @(Get-CimInstance Win32_Process -Filter "Name = 'ngrok.exe'" -ErrorAction Stop | Where-Object {
        -not [string]::IsNullOrWhiteSpace($_.CommandLine) -and
        $_.CommandLine -match '(?i)\bhttp\b.*\b8050\b'
    })

    foreach ($processInfo in $processes) {
        $processId = [int]$processInfo.ProcessId
        $process = Get-Process -Id $processId -ErrorAction SilentlyContinue
        if ($null -eq $process) {
            continue
        }

        Write-Host "Stopping dashboard ngrok tunnel (PID $processId)..."
        Stop-Process -Id $processId -ErrorAction Stop
        $process.WaitForExit(10000) | Out-Null
        if (-not $process.HasExited) {
            throw "Dashboard ngrok tunnel (PID $processId) did not stop within 10 seconds."
        }
    }

    return $processes.Count
}

function Start-AppProcess {
    param(
        [Parameter(Mandatory)][string]$Name,
        [Parameter(Mandatory)][string]$Module,
        [Parameter(Mandatory)][string]$StdoutLog,
        [Parameter(Mandatory)][string]$StderrLog,
        [string[]]$Arguments = @()
    )

    $process = Start-Process -FilePath $pythonExe `
        -ArgumentList (@("-m", $Module) + $Arguments) `
        -WorkingDirectory $repoRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $logRoot $StdoutLog) `
        -RedirectStandardError (Join-Path $logRoot $StderrLog) `
        -PassThru

    Write-Host "Started $Name (launcher PID $($process.Id))."
}

function Get-NgrokDashboardTunnel {
    try {
        $response = Invoke-RestMethod -Uri "http://127.0.0.1:4040/api/tunnels" -TimeoutSec 2
        return @($response.tunnels) | Where-Object {
            $_.config.addr -in @("http://localhost:8050", "http://127.0.0.1:8050")
        } | Select-Object -First 1
    }
    catch {
        return $null
    }
}

Push-Location $repoRoot
try {
    if ($IncludeIct -and $PrepareIctHandoff) {
        throw "-IncludeIct and -PrepareIctHandoff are mutually exclusive."
    }

    $esHeartbeat = Join-Path $healthRoot "es_shared_live_signal_service_heartbeat.json"
    if ($IncludeIct) {
        if (-not (Test-CleanIctHandoffSnapshot -HeartbeatPath $esHeartbeat)) {
            throw "-IncludeIct requires a healthy, non-stale, clean stopped ES heartbeat no more than 120 seconds old and no process at its recorded PID. No app process was stopped or started."
        }
        Write-Host "Running the final clean-handoff ICT readiness audit before restarting any app process..."
        & $pythonExe "scripts\audit_ict_paper_signal_readiness.py" "--allow-clean-stopped-handoff"
        if ($LASTEXITCODE -ne 0) {
            throw "ICT clean-handoff readiness is blocked. No app process was stopped or started. Use -PrepareIctHandoff after resolving substantive blockers."
        }
    }

    $stoppedProcessCount = 0
    $stoppedProcessCount += Stop-AppModuleProcesses `
        -Name "dashboard" `
        -Module "ote_live.scripts.run_live_dashboard"
    $stoppedProcessCount += Stop-AppModuleProcesses `
        -Name "shared FRVP/ICT collector" `
        -Module "ote_live.scripts.run_es_live_collector"
    $stoppedProcessCount += Stop-AppModuleProcesses `
        -Name "OTE collector" `
        -Module "ote_live.scripts.run_live_collector"
    $stoppedProcessCount += Stop-DashboardNgrokProcesses

    if ($stoppedProcessCount -gt 0) {
        Write-Host "Stopped $stoppedProcessCount existing app process(es); restarting the app."
        Start-Sleep -Milliseconds 500
    }
    else {
        Write-Host "No existing app processes were found; starting the app normally."
    }

    if (Test-LocalPort -Port 8050) {
        throw "Port 8050 is occupied by a process that was not identified as this app's dashboard."
    }
    else {
        Start-AppProcess `
            -Name "dashboard" `
            -Module "ote_live.scripts.run_live_dashboard" `
            -StdoutLog "live_dashboard_stdout.log" `
            -StderrLog "live_dashboard_stderr.log"
    }

    if (Test-HeartbeatProcess -HeartbeatPath $esHeartbeat) {
        Write-Host "Shared FRVP/ICT collector is already running."
    }
    else {
        $esCollectorArguments = @("--include-ict", "--allow-ict-clean-handoff")
        if ($PrepareIctHandoff) {
            $esCollectorArguments = @("--exclude-ict", "--max-cycles", "1")
            Write-Host "Starting a one-cycle FRVP-only ES bootstrap that must exit cleanly before the 120-second ICT handoff."
        }
        elseif (-not $IncludeIct) {
            $esCollectorArguments = @("--exclude-ict")
            Write-Host "Starting the shared ES collector in ongoing FRVP-only mode. Use -PrepareIctHandoff for the controlled ICT launch sequence."
        }
        Start-AppProcess `
            -Name "shared FRVP/ICT collector" `
            -Module "ote_live.scripts.run_es_live_collector" `
            -StdoutLog "es_shared_launcher_stdout.log" `
            -StderrLog "es_shared_launcher_stderr.log" `
            -Arguments $esCollectorArguments
    }

    $oteHeartbeat = Join-Path $healthRoot "live_signal_service_heartbeat.json"
    if (Test-HeartbeatProcess -HeartbeatPath $oteHeartbeat) {
        Write-Host "OTE collector is already running."
    }
    else {
        Start-AppProcess `
            -Name "OTE collector" `
            -Module "ote_live.scripts.run_live_collector" `
            -StdoutLog "ote_live_launcher_stdout.log" `
            -StderrLog "ote_live_launcher_stderr.log"
    }

    $tunnel = Get-NgrokDashboardTunnel
    if ($null -ne $tunnel) {
        Write-Host "Ngrok is already forwarding $($tunnel.public_url) to $($tunnel.config.addr)."
    }
    else {
        $runningNgrok = Get-Process -Name "ngrok" -ErrorAction SilentlyContinue
        if ($runningNgrok) {
            throw "Ngrok is running, but no tunnel to port 8050 was detected. Stop the stale ngrok process and run this script again."
        }

        $ngrokCommand = Get-Command "ngrok" -ErrorAction Stop
        $ngrokProcess = Start-Process -FilePath $ngrokCommand.Source `
            -ArgumentList @("http", "8050", "--url", $NgrokUrl, "--log", "stdout") `
            -WorkingDirectory $repoRoot `
            -WindowStyle Hidden `
            -RedirectStandardOutput (Join-Path $logRoot "ngrok_stdout.log") `
            -RedirectStandardError (Join-Path $logRoot "ngrok_stderr.log") `
            -PassThru

        Write-Host "Started ngrok (PID $($ngrokProcess.Id))."

        $tunnel = $null
        for ($attempt = 0; $attempt -lt 20 -and $null -eq $tunnel; $attempt++) {
            Start-Sleep -Milliseconds 500
            $tunnel = Get-NgrokDashboardTunnel
        }

        if ($null -eq $tunnel) {
            throw "Ngrok started, but its port 8050 tunnel was not ready within 10 seconds. Check $logRoot\ngrok_stderr.log"
        }

        Write-Host "Ngrok is forwarding $($tunnel.public_url) to $($tunnel.config.addr)."
    }
}
finally {
    Pop-Location
}
