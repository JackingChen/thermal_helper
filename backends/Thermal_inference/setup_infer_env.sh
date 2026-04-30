#!/usr/bin/env bash
# setup_infer_env.sh — one-shot environment setup for OMEN1526 inference
# Run once:  bash setup_infer_env.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/venv"

echo "=== [1/4] Creating venv at $VENV ==="
python3 -m venv "$VENV"

echo "=== [2/4] Upgrading pip ==="
"$VENV/bin/pip" install --upgrade pip -q

echo "=== [3/4] Installing PyTorch (CUDA 12.4 wheels — works with driver CUDA 13.0) ==="
"$VENV/bin/pip" install torch --index-url https://download.pytorch.org/whl/cu124 -q

echo "=== [4/4] Installing nvidia-physicsnemo-sym ==="
"$VENV/bin/pip" install nvidia-physicsnemo-sym numpy -q

echo ""
echo "=== Setup complete ==="
echo "Activate with:  source $VENV/bin/activate"
echo "Run inference:  python infer_thermal_solid.py"
