#!/usr/bin/env bash
# ForzaTek AI — launcher
# Starts the FastAPI backend on port 8000.
# Open http://localhost:8000 in your browser.

set -e
cd "$(dirname "$0")/.."
mkdir -p data models

if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

source .venv/bin/activate
pip install -q -r requirements.txt

echo "Starting ForzaTek AI on http://localhost:8000"
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
