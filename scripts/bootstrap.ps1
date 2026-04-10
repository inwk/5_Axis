param(
    [string]$EnvName = "ai_cam_5axis",
    [switch]$Recreate
)

$ErrorActionPreference = "Stop"

$setupScript = Join-Path $PSScriptRoot "setup_env.ps1"
$verifyScript = Join-Path $PSScriptRoot "verify_env.ps1"

if ($Recreate) {
    & $setupScript -EnvName $EnvName -Recreate
}
else {
    & $setupScript -EnvName $EnvName
}

& $verifyScript -EnvName $EnvName
