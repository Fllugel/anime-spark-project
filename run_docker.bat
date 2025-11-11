@echo off

echo Building PySpark Docker image...
docker build -f Dockerfile.uu -t my-spark .

if %errorlevel% neq 0 (
    echo.
    echo Docker build failed!
    pause
    exit /b %errorlevel%
)

echo.
echo Running PySpark container...
docker run --rm -v "%cd%":/app my-spark python main.py

echo.
echo Press any key to exit...
pause >nul
