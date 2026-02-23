#!/usr/bin/env bash
set -euo pipefail

# Sequential baseline -> GMP runs on TGV2D for SEGNN and GNS.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

EXTRA_ARGS=("$@")

echo "[1/4] SEGNN baseline"
python main.py config=configs/tgv_2d/segnn.yaml mode=all "${EXTRA_ARGS[@]}"

echo "[2/4] SEGNN + GMP"
python main.py config=configs/tgv_2d/segnn_gmp.yaml mode=all "${EXTRA_ARGS[@]}"

echo "[3/4] GNS baseline"
python main.py config=configs/tgv_2d/gns.yaml mode=all "${EXTRA_ARGS[@]}"

echo "[4/4] GNS + GMP"
python main.py config=configs/tgv_2d/gns_gmp.yaml mode=all "${EXTRA_ARGS[@]}"

echo "Done. Compare eval metrics under checkpoint/run directories."
