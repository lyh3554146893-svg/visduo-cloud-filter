@echo off
setlocal

echo ============================================================
echo VisDuo upload model - no Docker
echo ============================================================

set "CFG=%~dp0cloud_config.txt"
if not exist "%CFG%" (
    echo ERROR: cloud_config.txt not found.
    pause
    exit /b 1
)

for /f "usebackq tokens=1,* delims==" %%A in ("%CFG%") do (
    if /I "%%A"=="SERVER" set "SERVER=%%B"
    if /I "%%A"=="PORT" set "PORT=%%B"
    if /I "%%A"=="USER" set "USER=%%B"
)

if "%PORT%"=="" set "PORT=22"
if "%USER%"=="" set "USER=root"

if "%~1"=="" (
    echo Usage:
    echo cloud_upload_model.bat D:\models\best.pt
    pause
    exit /b 1
)

set "MODEL_FILE=%~1"
if not exist "%MODEL_FILE%" (
    echo ERROR: model file not found: %MODEL_FILE%
    pause
    exit /b 1
)

ssh -p %PORT% %USER%@%SERVER% "mkdir -p /root/autodl-tmp/visduo/model"
scp -P %PORT% "%MODEL_FILE%" %USER%@%SERVER%:/root/autodl-tmp/visduo/model/%~nx1
if errorlevel 1 (
    echo ERROR: model upload failed.
    pause
    exit /b 1
)

echo.
echo Model uploaded.
echo Cloud model path:
echo /root/autodl-tmp/visduo/model/%~nx1
pause
