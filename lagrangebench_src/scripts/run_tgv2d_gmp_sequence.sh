#!/usr/bin/env bash
set -euo pipefail

# Fair protocol for SEGNN: baseline first, then GMP sweep with matched budgets/seeds.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Runtime/config controls (override via env vars).
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
CKP_DIR="${CKP_DIR:-ckp}"
LOG_DIR="${LOG_DIR:-logs}"

BASE_CONFIG="${BASE_CONFIG:-configs/tgv_2d/segnn.yaml}"
GMP_CONFIG="${GMP_CONFIG:-configs/tgv_2d/segnn_gmp.yaml}"

SEEDS="${SEEDS:-0}"
STEP_MAX="${STEP_MAX:-300000}"
EVAL_STEPS="${EVAL_STEPS:-10000}"
TRAIN_EVAL_TRAJS="${TRAIN_EVAL_TRAJS:-50}"
GPU_ID="${GPU_ID:-0}"
DTYPE="${DTYPE:-float32}"
NUM_WORKERS="${NUM_WORKERS:-0}"

GMP_KERNEL="${GMP_KERNEL:-3}"
GMP_GRID_BINS_LIST="${GMP_GRID_BINS_LIST:-32 64 96}"
GMP_REDUCE_LIST="${GMP_REDUCE_LIST:-sum mean}"

EXTRA_ARGS=("$@")

ensure_runtime() {
  local smoke_check
  smoke_check='import importlib.util, sys; mods=["omegaconf","jax","haiku","e3nn_jax","jraph","optax","torch"]; missing=[m for m in mods if importlib.util.find_spec(m) is None]; print("runtime missing:", missing); sys.exit(1 if missing else 0)'

  echo "[preflight] python executable: ${PYTHON_BIN}"
  "${PYTHON_BIN}" -V

  if ! "${PYTHON_BIN}" -c "${smoke_check}"; then
    echo "[preflight] Installing runtime deps with ${PYTHON_BIN} -m pip"
    "${PYTHON_BIN}" -m pip install -U --no-cache-dir -r requirements_cuda.txt
    "${PYTHON_BIN}" -c "${smoke_check}"
  fi
}

ensure_runtime

mkdir -p "$CKP_DIR" "$LOG_DIR"

run_one() {
  local run_name="$1"
  shift

  echo "[run] ${run_name}"
  "${PYTHON_BIN}" -u main.py \
    mode=all \
    gpu="${GPU_ID}" \
    dtype="${DTYPE}" \
    train.num_workers="${NUM_WORKERS}" \
    train.step_max="${STEP_MAX}" \
    logging.eval_steps="${EVAL_STEPS}" \
    eval.train.n_trajs="${TRAIN_EVAL_TRAJS}" \
    eval.test=true \
    logging.wandb=false \
    logging.ckp_dir="${CKP_DIR}" \
    logging.run_name="${run_name}" \
    "$@" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "${LOG_DIR}/${run_name}.log"
}

echo "[phase 1/2] SEGNN fair baseline first"
for seed in ${SEEDS}; do
  run_one \
    "segnn_baseline_seed${seed}_${RUN_TAG}" \
    config="${BASE_CONFIG}" \
    seed="${seed}"
done

echo "[phase 2/2] SEGNN+GMP sweep with matched settings"
for reduce in ${GMP_REDUCE_LIST}; do
  for bins in ${GMP_GRID_BINS_LIST}; do
    for seed in ${SEEDS}; do
      run_one \
        "segnn_gmp_k${GMP_KERNEL}_b${bins}_${reduce}_seed${seed}_${RUN_TAG}" \
        config="${GMP_CONFIG}" \
        seed="${seed}" \
        model.use_gmp=true \
        model.gmp_kernel="${GMP_KERNEL}" \
        model.gmp_grid_bins="${bins}" \
        model.gmp_reduce="${reduce}"
    done
  done
done

echo "Done. Baseline completed first, then GMP sweep."
echo "Checkpoints: ${CKP_DIR}"
echo "Logs: ${LOG_DIR}"
