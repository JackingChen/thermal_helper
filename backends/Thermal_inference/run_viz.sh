#!/usr/bin/env bash
# run_viz.sh — Build the viz image (once) and launch the Streamlit UI.
#
# Usage:
#   bash run_viz.sh          # default port 8502
#   bash run_viz.sh 8888     # custom port
#
# Open http://localhost:<PORT> in your browser.
# The model and results.npy are written back to this folder via the volume mount.

set -e

PORT=${1:-8502}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_TAG="thermal-viz"

echo "==> Building Docker image '${IMAGE_TAG}' (only rebuilds if changed)…"
docker build \
  -f "${SCRIPT_DIR}/Dockerfile.viz" \
  -t "${IMAGE_TAG}" \
  "${SCRIPT_DIR}"

echo ""
echo "==> Starting Thermal Viz UI — open http://localhost:${PORT}"
echo "    Press Ctrl-C to stop."
echo ""

docker run --rm \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -p "${PORT}:8502" \
  -v "${SCRIPT_DIR}:/workspace" \
  -w /workspace \
  "${IMAGE_TAG}"

