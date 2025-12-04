@echo off
REM Simple helper script to run the Anime Spark app inside a Docker container with Spark.
REM It will check if the Docker image exists; if not, it will build it from the local Dockerfile.
REM
REM Default image name: my-spark-img
REM Usage:
REM   run_docker_windows.bat
REM   run_docker_windows.bat custom-image-name

setlocal enabledelayedexpansion

set IMAGE_NAME=my-spark-img
if NOT "%1"=="" (
    set IMAGE_NAME=%1
)

echo ================================================
echo Anime Spark - Docker launcher (image: %IMAGE_NAME%)
echo ================================================
echo.

REM Check if image exists; if not, build it
docker image inspect %IMAGE_NAME% >nul 2>&1
if errorlevel 1 (
    echo Docker image "%IMAGE_NAME%" not found. Building from local Dockerfile...
    docker build -t %IMAGE_NAME% .
    if errorlevel 1 (
        echo Failed to build Docker image "%IMAGE_NAME%".
        goto :end
    )
    echo.
)

echo Current directory will be mounted into /app inside the container.
echo.

docker run --rm -it ^
  -v "%cd%":/app ^
  -w /app ^
  %IMAGE_NAME% ^
  python main.py

:end
echo.
echo ================================================
echo Execution finished. Press any key to close...
echo ================================================
pause
endlocal


