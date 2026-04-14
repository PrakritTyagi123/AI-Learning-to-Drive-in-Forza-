#!/usr/bin/env bash
set -e
echo ""
echo "  FORZATEK AI SYSTEMS v2.0 (Phase 1 + 2)"
echo "  ========================================"
echo ""

if ! command -v python3 &> /dev/null; then echo "[ERROR] Python3 not found."; exit 1; fi

echo "[SETUP] Installing dependencies..."
pip3 install websockets mss opencv-python-headless numpy Pillow --quiet 2>/dev/null || pip3 install websockets mss opencv-python-headless numpy Pillow --quiet --break-system-packages
pip3 install torch torchvision --quiet 2>/dev/null || pip3 install torch torchvision --quiet --break-system-packages
pip3 install ultralytics --quiet 2>/dev/null || pip3 install ultralytics --quiet --break-system-packages

echo ""
echo "[START] Dashboard: http://localhost:8080"
echo "[START] Press Ctrl+C to stop."
echo ""

cd "$(dirname "$0")"
python3 backend/server.py
