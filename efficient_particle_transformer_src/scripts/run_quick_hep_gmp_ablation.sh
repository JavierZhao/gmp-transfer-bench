#!/usr/bin/env bash
set -euo pipefail

# Quick JetClass ablation: ParT vs ParTGMP under a small compute budget.
# Extra CLI args are forwarded to train_JetClass.sh.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SAMPLES_PER_EPOCH="${SAMPLES_PER_EPOCH:-32768}"
SAMPLES_PER_EPOCH_VAL="${SAMPLES_PER_EPOCH_VAL:-8192}"
NUM_EPOCHS="${NUM_EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-512}"
FEATURE_TYPE="${FEATURE_TYPE:-full}"
COMMENT_BASE="${COMMENT_BASE:-_quick_ablation}"

COMMON_ARGS=(
  --samples-per-epoch "$SAMPLES_PER_EPOCH"
  --samples-per-epoch-val "$SAMPLES_PER_EPOCH_VAL"
  --num-epochs "$NUM_EPOCHS"
  --batch-size "$BATCH_SIZE"
)

echo "[1/2] Running ParT baseline"
COMMENT="${COMMENT_BASE}_part" ./train_JetClass.sh ParT "$FEATURE_TYPE" "${COMMON_ARGS[@]}" "$@"

echo "[2/2] Running ParTGMP"
COMMENT="${COMMENT_BASE}_gmp" ./train_JetClass.sh ParTGMP "$FEATURE_TYPE" "${COMMON_ARGS[@]}" "$@"

echo "Done. Compare logs under logs/ and checkpoints under training/JetClass/."
