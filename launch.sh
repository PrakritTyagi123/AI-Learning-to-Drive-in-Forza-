#!/usr/bin/env bash
# ═══════════════════════════════════════════════
#  ForzaTek AI — Launch Script (Linux/Mac/WSL)
#  Installs dependencies and starts the server
# ═══════════════════════════════════════════════

set -e

echo ""
echo "  ╔══════════════════════════════════════════╗"
echo "  ║         FORZATEK AI SYSTEMS v1.0.4       ║"
echo "  ╚══════════════════════════════════════════╝"
echo ""

# ─── Check Python ───
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python3 not found. Install Python 3.10+"
    exit 1
fi

# ─── Install dependencies ───
echo "[SETUP] Installing Python dependencies..."
pip3 install websockets mss opencv-python-headless numpy Pillow --quiet 2>/dev/null || \
pip3 install websockets mss opencv-python-headless numpy Pillow --quiet --user 2>/dev/null || \
pip3 install websockets mss opencv-python-headless numpy Pillow --quiet --break-system-packages 2>/dev/null

echo ""
echo "[SETUP] Dependencies installed."
echo ""

# ─── Launch ───
echo "[START] Launching ForzaTek AI..."
echo "[START] Dashboard: http://localhost:8080"
echo "[START] Press Ctrl+C to stop."
echo ""

cd "$(dirname "$0")"
python3 backend/server.py
