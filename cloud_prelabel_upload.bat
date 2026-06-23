@echo off
setlocal

echo ============================================================
echo VisDuo YOLO prelabel - upload local data, no Docker
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
    if /I "%%A"=="KEEP_CLOUD" set "KEEP_CLOUD=%%B"
)

if "%PORT%"=="" set "PORT=22"
if "%USER%"=="" set "USER=root"
if "%KEEP_CLOUD%"=="" set "KEEP_CLOUD=0"

if "%~1"=="" (
    echo Usage:
    echo cloud_prelabel_upload.bat ^<local_filter_output_or_image_dir^> [prelabel_args]
    echo.
    echo Examples:
    echo cloud_prelabel_upload.bat D:\visduo_cloud_filter_clean\cloud_output_xxx --model yolov8n.pt --save_preview
    echo cloud_prelabel_upload.bat D:\visduo_cloud_filter_clean\cloud_output_xxx --model /root/autodl-tmp/visduo/model/best.pt --keep_names face phone smoke --save_preview
    pause
    exit /b 1
)

set "LOCAL_INPUT=%~1"
shift
set "PRELABEL_ARGS=%*"

if not exist "%LOCAL_INPUT%" (
    echo ERROR: local input not found: %LOCAL_INPUT%
    pause
    exit /b 1
)

set "DATESTR=%date:~0,4%%date:~5,2%%date:~8,2%"
set "TIMESTR=%time:~0,2%%time:~3,2%%time:~6,2%"
set "TASK_ID=%DATESTR%_%TIMESTR%"
set "TASK_ID=%TASK_ID: =0%"

set "CLOUD_INPUT=/root/autodl-tmp/visduo/tasks/%TASK_ID%/prelabel_input"
set "CLOUD_OUT=/root/autodl-tmp/visduo/tasks/%TASK_ID%/prelabel_output"
set "LOCAL_OUT=%~dp0prelabel_output_%TASK_ID%"

echo SERVER     : %USER%@%SERVER%:%PORT%
echo LOCAL INPUT: %LOCAL_INPUT%
echo LOCAL OUT  : %LOCAL_OUT%
echo ARGS       : %PRELABEL_ARGS%
echo.

echo [1/3] Upload input to cloud...
ssh -p %PORT% %USER%@%SERVER% "rm -rf %CLOUD_INPUT% %CLOUD_OUT%; mkdir -p %CLOUD_INPUT% %CLOUD_OUT%"
if errorlevel 1 (
    echo ERROR: cloud folder prepare failed.
    pause
    exit /b 1
)

scp -P %PORT% -r "%LOCAL_INPUT%\*" %USER%@%SERVER%:%CLOUD_INPUT%/
if errorlevel 1 (
    echo ERROR: input upload failed.
    pause
    exit /b 1
)

echo.
echo [2/3] Run YOLO prelabel on cloud...
ssh -p %PORT% %USER%@%SERVER% "cd /root/autodl-tmp/visduo/app && /root/autodl-tmp/visduo/venv/bin/python prelabel_yolo.py --input %CLOUD_INPUT% --output_dir %CLOUD_OUT% %PRELABEL_ARGS%"
if errorlevel 1 (
    echo ERROR: cloud prelabel failed.
    pause
    exit /b 1
)

echo.
echo [3/3] Download prelabel results...
mkdir "%LOCAL_OUT%" 2>nul
scp -P %PORT% -r %USER%@%SERVER%:%CLOUD_OUT%/* "%LOCAL_OUT%\"
if errorlevel 1 (
    echo ERROR: result download failed.
    pause
    exit /b 1
)

if "%KEEP_CLOUD%"=="0" (
    ssh -p %PORT% %USER%@%SERVER% "rm -rf /root/autodl-tmp/visduo/tasks/%TASK_ID%"
)

echo.
echo Prelabel OK.
echo Local output:
echo %LOCAL_OUT%
start "" "%LOCAL_OUT%"
pause
