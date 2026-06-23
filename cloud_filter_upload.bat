@echo off
setlocal

echo ============================================================
echo VisDuo cloud filter - upload local data, no Docker
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
    echo cloud_filter_upload.bat ^<local_data_dir^> [filter_args]
    echo.
    echo Examples:
    echo cloud_filter_upload.bat D:\data --disable_yolo --enable_copyright --enable_ocr --copy_mode all
    echo cloud_filter_upload.bat D:\data --expected person 'cell phone' --enable_copyright --copy_mode all
    pause
    exit /b 1
)

set "DATA_DIR=%~1"
shift
set "FILTER_ARGS=%*"

if not exist "%DATA_DIR%" (
    echo ERROR: local data dir not found: %DATA_DIR%
    pause
    exit /b 1
)

set "DATESTR=%date:~0,4%%date:~5,2%%date:~8,2%"
set "TIMESTR=%time:~0,2%%time:~3,2%%time:~6,2%"
set "TASK_ID=%DATESTR%_%TIMESTR%"
set "TASK_ID=%TASK_ID: =0%"

set "CLOUD_DATA=/root/autodl-tmp/visduo/tasks/%TASK_ID%/data"
set "CLOUD_OUT=/root/autodl-tmp/visduo/tasks/%TASK_ID%/output"
set "LOCAL_OUT=%~dp0cloud_output_%TASK_ID%"

echo SERVER    : %USER%@%SERVER%:%PORT%
echo LOCAL DATA: %DATA_DIR%
echo LOCAL OUT : %LOCAL_OUT%
echo ARGS      : %FILTER_ARGS%
echo.

echo [1/3] Upload local data to cloud...
ssh -p %PORT% %USER%@%SERVER% "rm -rf %CLOUD_DATA% %CLOUD_OUT%; mkdir -p %CLOUD_DATA% %CLOUD_OUT%"
if errorlevel 1 (
    echo ERROR: cloud folder prepare failed.
    pause
    exit /b 1
)

scp -P %PORT% -r "%DATA_DIR%\*" %USER%@%SERVER%:%CLOUD_DATA%/
if errorlevel 1 (
    echo ERROR: data upload failed.
    pause
    exit /b 1
)

echo.
echo [2/3] Run filter on cloud...
ssh -p %PORT% %USER%@%SERVER% "cd /root/autodl-tmp/visduo/app && /root/autodl-tmp/visduo/venv/bin/python visual_filter.py --input %CLOUD_DATA% --output_dir %CLOUD_OUT% %FILTER_ARGS%"
if errorlevel 1 (
    echo ERROR: cloud filter failed.
    pause
    exit /b 1
)

echo.
echo [3/3] Download results to local...
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
echo Filter OK.
echo Local output:
echo %LOCAL_OUT%
start "" "%LOCAL_OUT%"
pause
