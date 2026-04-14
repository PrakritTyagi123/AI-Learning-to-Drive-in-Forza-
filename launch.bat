@echo off
REM ═══════════════════════════════════════════════
REM  ForzaTek AI v2.0 — Launch Script (Windows)
REM  Phase 1 (Dashboard) + Phase 2 (AI Vision)
REM ═══════════════════════════════════════════════

echo.
echo   FORZATEK AI SYSTEMS v2.0 (Phase 1 + 2)
echo   ========================================
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Install Python 3.10+ from python.org
    pause
    exit /b 1
)

echo [SETUP] Installing dependencies...
echo [SETUP] This may take a few minutes on first run (PyTorch is ~2GB)...
pip install websockets mss opencv-python-headless numpy Pillow --quiet --disable-pip-version-check
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --quiet --disable-pip-version-check
pip install ultralytics --quiet --disable-pip-version-check

echo.
echo [SETUP] Dependencies installed.
echo.
echo [START] Launching ForzaTek AI...
echo [START] Dashboard: http://localhost:8080
echo [START] Press Ctrl+C to stop.
echo.

cd /d "%~dp0"
python backend\server.py

pause
