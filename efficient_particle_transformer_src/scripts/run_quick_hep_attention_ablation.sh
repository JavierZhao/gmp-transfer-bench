#!/usr/bin/env bash
set -euo pipefail

# Quick JetClass ablation with matched budgets across:
#   ParT, ParTGMP, ParTGPCA, ParTSADA
# Extra CLI args are forwarded to train_JetClass.sh.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SAMPLES_PER_EPOCH="${SAMPLES_PER_EPOCH:-32768}"
SAMPLES_PER_EPOCH_VAL="${SAMPLES_PER_EPOCH_VAL:-8192}"
NUM_EPOCHS="${NUM_EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-512}"
FEATURE_TYPE="${FEATURE_TYPE:-full}"
COMMENT_BASE="${COMMENT_BASE:-_quick_attention_ablation}"
ABLATION_MODELS="${ABLATION_MODELS:-ParT ParTGMP ParTGPCA ParTSADA}"

read -r -a MODELS <<< "$ABLATION_MODELS"

COMMON_ARGS=(
  --samples-per-epoch "$SAMPLES_PER_EPOCH"
  --samples-per-epoch-val "$SAMPLES_PER_EPOCH_VAL"
  --num-epochs "$NUM_EPOCHS"
  --batch-size "$BATCH_SIZE"
)

comment_suffix() {
  case "$1" in
    ParT) echo "_part" ;;
    ParTGMP) echo "_gmp" ;;
    ParTGPCA) echo "_gpca" ;;
    ParTSADA) echo "_sada" ;;
    *) echo "_$(echo "$1" | tr '[:upper:]' '[:lower:]')" ;;
  esac
}

num_models="${#MODELS[@]}"
for i in "${!MODELS[@]}"; do
  model="${MODELS[$i]}"
  suffix="$(comment_suffix "$model")"
  echo "[$((i + 1))/${num_models}] Running ${model}"
  COMMENT="${COMMENT_BASE}${suffix}" ./train_JetClass.sh "$model" "$FEATURE_TYPE" "${COMMON_ARGS[@]}" "$@"
done

echo "Done. Compare logs under logs/ and checkpoints under training/JetClass/."
