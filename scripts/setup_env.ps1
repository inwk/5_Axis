param(
    [string]$EnvName = "ai_cam_5axis",
    [string]$EnvironmentFile = "environment.yml",
    [switch]$Recreate
)

$ErrorActionPreference = "Stop"

function Resolve-CondaPath {
    $candidates = @(
        "C:\Users\inwoo\anaconda3\Scripts\conda.exe",
        "C:\Users\inwoo\miniconda3\Scripts\conda.exe",
        "C:\ProgramData\anaconda3\Scripts\conda.exe",
        "C:\ProgramData\miniconda3\Scripts\conda.exe"
    )
    foreach ($path in $candidates) {
        if (Test-Path $path) {
            return $path
        }
    }

    $cmd = Get-Command conda -ErrorAction SilentlyContinue
    if ($cmd -and $cmd.Source) {
        return $cmd.Source
    }
    throw "conda.exe not found. Install Anaconda/Miniconda first."
}

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

if (-not (Test-Path $EnvironmentFile)) {
    throw "Environment file not found: $EnvironmentFile"
}

$conda = Resolve-CondaPath
$env:CONDA_NO_PLUGINS = "true"

if ($Recreate) {
    Write-Host "[setup] Removing existing env: $EnvName"
    & $conda env remove -n $EnvName -y | Out-Host
}

$envListText = (& $conda env list 2>$null | Out-String)
if ($envListText -match "(^|\s)$([regex]::Escape($EnvName))(\s|$)") {
    Write-Host "[setup] Updating env: $EnvName"
    & $conda env update -n $EnvName -f $EnvironmentFile --prune | Out-Host
}
else {
    Write-Host "[setup] Creating env: $EnvName"
    & $conda env create -n $EnvName -f $EnvironmentFile | Out-Host
}

Write-Host "[setup] Upgrading pip in env"
& $conda run -n $EnvName python -m pip install --upgrade pip | Out-Host

Write-Host "[setup] Done. Run: scripts\verify_env.ps1 -EnvName $EnvName"
