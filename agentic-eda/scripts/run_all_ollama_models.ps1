param(
    [string[]]$Models = @(
        "phi3:mini",
        "mistral:latest",
        "tinyllama:latest",
        "smollm2:1.7b",
        "gemma2:2b",
        "qwen2.5:7b",
        "phi4:latest",
        "llama3.2:3b",
        "qwen2.5:3b",
        "llama3.1:8b"
    ),
    [string]$PythonExe = "",
    [string]$LlmProvider = "ollama",
    [switch]$FreshComparison,
    [switch]$SkipWarmup,
    [int]$PauseSeconds = 2,
    [switch]$KeepLastModelLoaded
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Resolve-Path (Join-Path $scriptDir "..")
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logDir = Join-Path $projectRoot "outputs\batch_logs"
$logPath = Join-Path $logDir "run_all_models_$timestamp.log"

New-Item -Path $logDir -ItemType Directory -Force | Out-Null

function Write-Log {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Write-Host $line
    Add-Content -LiteralPath $logPath -Value $line
}

function Resolve-PythonExecutable {
    param([string]$Preferred)

    $candidates = @()
    if (-not [string]::IsNullOrWhiteSpace($Preferred)) {
        $candidates += $Preferred
    }
    $candidates += @(
        "python",
        "py",
        "C:\Users\Admin\miniconda3\python.exe"
    )

    foreach ($candidate in $candidates | Select-Object -Unique) {
        try {
            $cmd = Get-Command $candidate -ErrorAction SilentlyContinue
            if (-not $cmd) {
                continue
            }
            & $candidate "--version" *> $null
            if ($LASTEXITCODE -eq 0) {
                return $candidate
            }
        }
        catch {
            continue
        }
    }

    throw "Could not find a working Python executable. Pass -PythonExe with a full path."
}

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)][scriptblock]$Command,
        [Parameter(Mandatory = $true)][string]$Description
    )
    Write-Log $Description
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed (exit code $LASTEXITCODE): $Description"
    }
}

function Get-RunningModelNames {
    $output = & ollama ps 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to run 'ollama ps'. Ensure Ollama is running."
    }

    $lines = @($output)
    if ($lines.Count -le 1) {
        return @()
    }

    $names = @()
    foreach ($line in ($lines | Select-Object -Skip 1)) {
        if ([string]::IsNullOrWhiteSpace($line)) {
            continue
        }
        $parts = ($line -split "\s+") | Where-Object { $_ -ne "" }
        if ($parts.Count -ge 1) {
            $names += $parts[0]
        }
    }

    return @($names | Sort-Object -Unique)
}

function Stop-RunningModels {
    $running = @(Get-RunningModelNames)
    if ($running.Length -eq 0) {
        Write-Log "No currently running Ollama models."
        return
    }

    foreach ($model in $running) {
        Write-Log "Stopping running model: $model"
        & ollama stop $model | Out-Null
        if ($LASTEXITCODE -ne 0) {
            Write-Log "Warning: could not stop model $model (it may already be unloaded)."
        }
    }
}

function Start-ModelWarmup {
    param([Parameter(Mandatory = $true)][string]$Model)

    $payload = @{
        model = $Model
        messages = @(
            @{
                role = "user"
                content = "Reply with OK."
            }
        )
        stream = $false
        options = @{
            temperature = 0
        }
        keep_alive = "15m"
    } | ConvertTo-Json -Depth 6

    try {
        $null = Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:11434/api/chat" -Body $payload -ContentType "application/json" -TimeoutSec 120
        Write-Log "Model warmup succeeded: $Model"
    }
    catch {
        Write-Log "Warning: warmup failed for $Model. Continuing; run_all.py will still attempt to use it."
    }
}

Push-Location $projectRoot
try {
    $resolvedPython = Resolve-PythonExecutable -Preferred $PythonExe
    Write-Log "Project root: $projectRoot"
    Write-Log "Log file: $logPath"
    Write-Log ("Model count: {0}" -f $Models.Count)
    Write-Log "Python executable: $resolvedPython"

    if ($FreshComparison) {
        Write-Log "FreshComparison enabled: clearing aggregate comparison artifacts."
        Remove-Item "outputs\evaluation_results.csv" -Force -ErrorAction SilentlyContinue
        Remove-Item "outputs\comparison_summary.md" -Force -ErrorAction SilentlyContinue
        Remove-Item "outputs\statistical_significance_summary.md" -Force -ErrorAction SilentlyContinue
        Remove-Item "outputs\comparison_tables\*" -Recurse -Force -ErrorAction SilentlyContinue
        Remove-Item "outputs\comparison_plots\*" -Recurse -Force -ErrorAction SilentlyContinue
    }

    $failures = @()

    foreach ($model in $Models) {
        Write-Log ("=" * 72)
        Write-Log "Switching to model: $model"

        Stop-RunningModels
        Start-Sleep -Seconds $PauseSeconds

        if (-not $SkipWarmup) {
            Start-ModelWarmup -Model $model
        }

        $commandDescription = "Running evaluation with model $model"
        try {
            Invoke-Checked -Description $commandDescription -Command {
                & $resolvedPython "run_all.py" "--llm-provider" $LlmProvider "--llm-model" $model
            }
            Write-Log "Completed model: $model"
        }
        catch {
            $message = $_.Exception.Message
            Write-Log "ERROR for ${model}: $message"
            $failures += $model
        }
    }

    Write-Log ("=" * 72)
    if (-not $KeepLastModelLoaded) {
        Write-Log "Stopping active model after benchmark completion."
        Stop-RunningModels
    }

    Write-Log ("=" * 72)
    if ($failures.Count -eq 0) {
        Write-Log "All model runs completed successfully."
        exit 0
    }
    else {
        Write-Log ("Completed with failures for {0} model(s): {1}" -f $failures.Count, ($failures -join ", "))
        exit 1
    }
}
finally {
    Pop-Location
}
