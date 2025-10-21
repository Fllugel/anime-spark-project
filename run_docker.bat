@echo off

echo Running PySpark container...
docker run --rm -v "%cd%":/app my-spark python main.py

echo.
echo Press any key to exit...
pause >nul
