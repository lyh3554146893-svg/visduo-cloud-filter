@echo off
setlocal

echo ============================================================
echo VisDuo cloud deploy - no Docker version
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

if not exist "%~dp0visual_filter.py" (
    echo ERROR: visual_filter.py not found in this folder.
    pause
    exit /b 1
)

echo SERVER: %USER%@%SERVER%:%PORT%
echo CLOUD : /root/autodl-tmp/visduo
echo.

echo [1/4] Create cloud folders...
ssh -p %PORT% %USER%@%SERVER% "mkdir -p /root/autodl-tmp/visduo/app /root/autodl-tmp/visduo/tasks /root/autodl-tmp/visduo/model /root/autodl-tmp/visduo/logo_templates /root/autodl-tmp/visduo/copyright_refs"
if errorlevel 1 (
    echo ERROR: SSH failed. Check server, port, user, password.
    pause
    exit /b 1
)

echo.
echo [2/4] Upload scripts...
scp -P %PORT% "%~dp0visual_filter.py" "%~dp0prelabel_yolo.py" "%~dp0requirements.txt" %USER%@%SERVER%:/root/autodl-tmp/visduo/app/
if errorlevel 1 (
    echo ERROR: upload failed.
    pause
    exit /b 1
)

echo.
echo [3/4] Install system packages...
ssh -p %PORT% %USER%@%SERVER% "apt update && apt install -y python3-pip python3-venv ffmpeg tesseract-ocr tesseract-ocr-chi-sim tesseract-ocr-eng"
if errorlevel 1 (
    echo WARNING: apt install failed or partially failed.
    echo If packages already exist, you may continue. Otherwise check the cloud error.
)

echo.
echo [4/4] Create Python venv and install Python packages...
ssh -p %PORT% %USER%@%SERVER% "python3 -m venv /root/autodl-tmp/visduo/venv || true; /root/autodl-tmp/visduo/venv/bin/python -m pip install --upgrade pip; /root/autodl-tmp/visduo/venv/bin/pip install -r /root/autodl-tmp/visduo/app/requirements.txt"
if errorlevel 1 (
    echo ERROR: Python package install failed.
    pause
    exit /b 1
)

echo.
echo Deploy OK.
echo Next command:
echo cloud_filter_upload.bat D:\data --disable_yolo --enable_copyright --enable_ocr --copy_mode all
pause
