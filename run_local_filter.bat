@echo off
setlocal

echo ============================================================
echo VisDuo local filter
echo ============================================================

if "%~1"=="" (
    echo Usage:
    echo run_local_filter.bat ^<local_data_dir^> [filter_args]
    echo.
    echo Example:
    echo run_local_filter.bat D:\data --disable_yolo --enable_copyright --enable_ocr --copy_mode all
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

set "LOCAL_OUT=%~dp0local_output_%TASK_ID%"

python "%~dp0visual_filter.py" --input "%DATA_DIR%" --output_dir "%LOCAL_OUT%" %FILTER_ARGS%
if errorlevel 1 (
    echo ERROR: local filter failed.
    pause
    exit /b 1
)

echo.
echo Local filter OK.
echo Output:
echo %LOCAL_OUT%
start "" "%LOCAL_OUT%"
pause
