@echo off
setlocal

echo ============================================================
echo VisDuo cloud filter - server path, no upload, no Docker
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
    echo cloud_filter_server_path.bat ^<cloud_data_path^> [filter_args]
    echo.
    echo Example:
    echo cloud_filter_server_path.bat /root/autodl-tmp/data --disable_yolo --enable_copyright --enable_ocr --copy_mode all
    pause
    exit /b 1
)

set "CLOUD_DATA=%~1"
shift
set "FILTER_ARGS=%*"

set "DATESTR=%date:~0,4%%date:~5,2%%date:~8,2%"
set "TIMESTR=%time:~0,2%%time:~3,2%%time:~6,2%"
set "TASK_ID=%DATESTR%_%TIMESTR%"
set "TASK_ID=%TASK_ID: =0%"

set "CLOUD_OUT=/root/autodl-tmp/visduo/tasks/%TASK_ID%/output"
set "LOCAL_OUT=%~dp0cloud_output_%TASK_ID%"

echo SERVER    : %USER%@%SERVER%:%PORT%
echo CLOUD DATA: %CLOUD_DATA%
echo LOCAL OUT : %LOCAL_OUT%
echo ARGS      : %FILTER_ARGS%
echo.

echo [1/2] Run filter on cloud server path...
ssh -p %PORT% %USER%@%SERVER% "rm -rf %CLOUD_OUT%; mkdir -p %CLOUD_OUT%; cd /root/autodl-tmp/visduo/app && /root/autodl-tmp/visduo/venv/bin/python visual_filter.py --input %CLOUD_DATA% --output_dir %CLOUD_OUT% %FILTER_ARGS%"
if errorlevel 1 (
    echo ERROR: cloud filter failed.
    pause
    exit /b 1
)

echo.
echo [2/2] Download results to local...
mkdir "%LOCAL_OUT%" 2>nul
scp -P %PORT% -r %USER%@%SERVER%:%CLOUD_OUT%/* "%LOCAL_OUT%\"
if errorlevel 1 (
    echo ERROR: result download failed.
    pause
    exit /b 1
)

echo.
echo Filter OK.
echo Local output:
echo %LOCAL_OUT%
start "" "%LOCAL_OUT%"
pause
