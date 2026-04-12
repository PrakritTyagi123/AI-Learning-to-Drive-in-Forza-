@echo off
REM ═══════════════════════════════════════════════
REM  ForzaTek AI — Launch Script (Windows)
REM  Installs dependencies and starts the server
REM ═══════════════════════════════════════════════

echo.
echo   ╔══════════════════════════════════════════╗
echo   ║         FORZATEK AI SYSTEMS v1.0.4       ║
echo   ╚══════════════════════════════════════════╝
echo.

REM ─── Check Python ───
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Install Python 3.10+ from python.org
    pause
    exit /b 1
)

REM ─── Install dependencies ───
echo [SETUP] Installing Python dependencies...
pip install websockets mss opencv-python-headless numpy Pillow --quiet --disable-pip-version-check
if errorlevel 1 (
    echo [WARN] Some packages may have failed. Trying with --user flag...
    pip install websockets mss opencv-python-headless numpy Pillow --quiet --user --disable-pip-version-check
)

echo.
echo [SETUP] Dependencies installed.
echo.

REM ─── Launch server ───
echo [START] Launching ForzaTek AI...
echo [START] Dashboard will be at: http://localhost:8080
echo [START] Press Ctrl+C to stop.
echo.

cd /d "%~dp0"
python backend\server.py

pause
