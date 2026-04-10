param(
    [string]$EnvName = "ai_cam_5axis"
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
    throw "conda.exe not found."
}

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$conda = Resolve-CondaPath
$env:CONDA_NO_PLUGINS = "true"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

Write-Host "=== Environment Verification ==="

$verifyCode = @'
import importlib.util
import os
import runpy
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

print("=== Python/Torch ===")
print(sys.executable)
import torch
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())

print("=== Project Imports ===")
import graph_sdf
print("graph_sdf ok")

print("=== NXOpen Check ===")
spec = importlib.util.find_spec("NXOpen")
if spec:
    print("NXOpen module found")
else:
    print("NXOpen module NOT found (expected if NX runtime not installed/configured)")

print("=== Smoke Test ===")
runpy.run_path("smoke_test_graph_sdf.py", run_name="__main__")
'@

$tmpScript = Join-Path $PSScriptRoot "_tmp_verify_env.py"
Set-Content -Path $tmpScript -Value $verifyCode -Encoding UTF8

try {
    & $conda run -n $EnvName python $tmpScript
}
finally {
    if (Test-Path $tmpScript) {
        Remove-Item $tmpScript -Force
    }
}

Write-Host "[verify] complete"
