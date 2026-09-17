[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [int]$DashboardPort = 8050,
    [int]$TimeoutSeconds = 10,
    [switch]$SkipNgrok,
    [switch]$IncludeAnyOteLivePython
)

$ErrorActionPreference = "Stop"

if ($TimeoutSeconds -lt 1) {
    throw "TimeoutSeconds must be at least 1."
}

$repoRoot = $PSScriptRoot
$timeoutMilliseconds = [Math]::Max(1000, $TimeoutSeconds * 1000)
$requestedWhatIf = [bool]$WhatIfPreference

function Invoke-WithoutWhatIf {
    param([Parameter(Mandatory)][scriptblock]$ScriptBlock)

    $previousWhatIfPreference = $WhatIfPreference
    $WhatIfPreference = $false
    try {
        & $ScriptBlock
    }
    finally {
        $WhatIfPreference = $previousWhatIfPreference
    }
}

function Add-TargetProcess {
    param(
        [Parameter(Mandatory)][hashtable]$Targets,
        [Parameter(Mandatory)][object]$ProcessInfo,
        [Parameter(Mandatory)][string]$Name
    )

    $processId = [int]$ProcessInfo.ProcessId
    if (-not $Targets.ContainsKey($processId)) {
        $Targets[$processId] = [pscustomobject]@{
            ProcessId = $processId
            Name = $Name
            ProcessName = [string]$ProcessInfo.Name
            CommandLine = [string]$ProcessInfo.CommandLine
        }
    }
}

function Add-MatchingProcesses {
    param(
        [Parameter(Mandatory)][hashtable]$Targets,
        [Parameter(Mandatory)][object[]]$ProcessSnapshot,
        [Parameter(Mandatory)][string]$Name,
        [Parameter(Mandatory)][string]$ProcessNameRegex,
        [Parameter(Mandatory)][string]$CommandLineRegex
    )

    foreach ($processInfo in $ProcessSnapshot) {
        if ([string]::IsNullOrWhiteSpace($processInfo.CommandLine)) {
            continue
        }
        if ($processInfo.Name -notmatch $ProcessNameRegex) {
            continue
        }
        if ($processInfo.CommandLine -notmatch $CommandLineRegex) {
            continue
        }

        Add-TargetProcess -Targets $Targets -ProcessInfo $processInfo -Name $Name
    }
}

function Stop-TargetProcess {
    param([Parameter(Mandatory)]$Target)

    $process = Get-Process -Id $Target.ProcessId -ErrorAction SilentlyContinue
    if ($null -eq $process) {
        return $false
    }

    $targetLabel = "$($Target.Name) (PID $($Target.ProcessId))"
    if (-not $PSCmdlet.ShouldProcess($targetLabel, "Stop process")) {
        return $true
    }

    Write-Host "Stopping $targetLabel..."
    Stop-Process -Id $Target.ProcessId -ErrorAction Stop
    if (-not $process.WaitForExit($timeoutMilliseconds)) {
        throw "$targetLabel did not stop within $TimeoutSeconds second(s)."
    }

    return $true
}

Push-Location $repoRoot
try {
    $processSnapshot = @(Invoke-WithoutWhatIf { Get-CimInstance Win32_Process -ErrorAction Stop })
    $targets = @{}

    $knownProcessSpecs = @(
        @{
            Name = "OTE live signal supervisor"
            ProcessNameRegex = '^(powershell|pwsh)\.exe$'
            CommandLineRegex = '(?i)\brun_live_signal_service\.ps1\b'
        },
        @{
            Name = "OTE live stack wrapper"
            ProcessNameRegex = '^(powershell|pwsh)\.exe$'
            CommandLineRegex = '(?i)\bstart_ote_live_stack\.ps1\b'
        },
        @{
            Name = "OTE live stack launcher"
            ProcessNameRegex = '^pythonw?\.exe$'
            CommandLineRegex = '(?i)(ote_live\.scripts\.run_live_stack|ote_live[\\/]+scripts[\\/]+run_live_stack(?:\.py)?)'
        },
        @{
            Name = "dashboard"
            ProcessNameRegex = '^pythonw?\.exe$'
            CommandLineRegex = '(?i)(ote_live\.scripts\.run_live_dashboard|ote_live[\\/]+scripts[\\/]+run_live_dashboard(?:\.py)?)'
        },
        @{
            Name = "shared FRVP/ICT collector"
            ProcessNameRegex = '^pythonw?\.exe$'
            CommandLineRegex = '(?i)(ote_live\.scripts\.run_es_live_collector|ote_live[\\/]+scripts[\\/]+run_es_live_collector(?:\.py)?)'
        },
        @{
            Name = "FRVP live collector"
            ProcessNameRegex = '^pythonw?\.exe$'
            CommandLineRegex = '(?i)(ote_live\.scripts\.run_frvp_live_collector|ote_live[\\/]+scripts[\\/]+run_frvp_live_collector(?:\.py)?)'
        },
        @{
            Name = "OTE collector"
            ProcessNameRegex = '^pythonw?\.exe$'
            CommandLineRegex = '(?i)(ote_live\.scripts\.run_live_collector|ote_live[\\/]+scripts[\\/]+run_live_collector(?:\.py)?)'
        }
    )

    foreach ($spec in $knownProcessSpecs) {
        Add-MatchingProcesses `
            -Targets $targets `
            -ProcessSnapshot $processSnapshot `
            -Name $spec.Name `
            -ProcessNameRegex $spec.ProcessNameRegex `
            -CommandLineRegex $spec.CommandLineRegex
    }

    if ($IncludeAnyOteLivePython) {
        $escapedRepoOteLive = [regex]::Escape((Join-Path $repoRoot "ote_live"))
        Add-MatchingProcesses `
            -Targets $targets `
            -ProcessSnapshot $processSnapshot `
            -Name "OTE live Python process" `
            -ProcessNameRegex '^pythonw?\.exe$' `
            -CommandLineRegex "(?i)(\bote_live\.|$escapedRepoOteLive)"
    }

    if (-not $SkipNgrok) {
        Add-MatchingProcesses `
            -Targets $targets `
            -ProcessSnapshot $processSnapshot `
            -Name "dashboard ngrok tunnel" `
            -ProcessNameRegex '^ngrok\.exe$' `
            -CommandLineRegex "(?i)\bhttp\b.*\b$DashboardPort\b"
    }

    $orderedTargets = @(
        $targets.Values |
            Sort-Object `
                @{ Expression = {
                    switch -Regex ($_.Name) {
                        "supervisor|wrapper|launcher" { 0; break }
                        "collector" { 1; break }
                        "dashboard" { 2; break }
                        "ngrok" { 3; break }
                        default { 4 }
                    }
                } },
                @{ Expression = "ProcessId" }
    )

    if ($orderedTargets.Count -eq 0) {
        Write-Host "No running OTE live processes were found."
        return
    }

    $stoppedProcessCount = 0
    $failures = @()
    foreach ($target in $orderedTargets) {
        try {
            if (Stop-TargetProcess -Target $target) {
                $stoppedProcessCount += 1
            }
        }
        catch {
            $failures += "$($target.Name) (PID $($target.ProcessId)): $($_.Exception.Message)"
            Write-Warning $failures[-1]
        }
    }

    if ($failures.Count -gt 0) {
        throw "Failed to stop $($failures.Count) OTE live process(es)."
    }

    if ($requestedWhatIf) {
        Write-Host "Would stop $stoppedProcessCount OTE live process(es)."
    }
    else {
        Write-Host "Stopped $stoppedProcessCount OTE live process(es)."
    }
}
finally {
    Pop-Location
}
