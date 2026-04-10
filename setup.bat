@echo off
setlocal

set SCRIPT_DIR=%~dp0
powershell -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%scripts\bootstrap.ps1" %*
set EXIT_CODE=%ERRORLEVEL%

if not "%EXIT_CODE%"=="0" (
    echo [setup] failed with exit code %EXIT_CODE%
    exit /b %EXIT_CODE%
)

echo [setup] complete
exit /b 0
