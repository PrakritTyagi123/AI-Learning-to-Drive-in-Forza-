@echo off
REM ForzaTek AI -- launcher for Windows
REM Starts the FastAPI backend on port 8000.
REM Open http://localhost:8000 in your browser.

cd /d "%~dp0\.."
if not exist data mkdir data
if not exist models mkdir models

if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
)

call .venv\Scripts\activate.bat
pip install -q -r requirements.txt

echo Starting ForzaTek AI on http://localhost:8000
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
