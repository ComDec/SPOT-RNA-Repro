#!/usr/bin/env bash
set -euo pipefail

GPU_A="${1:-0}"
GPU_B="${2:-1}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-5}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-10}"
NUM_WORKERS="${NUM_WORKERS:-2}"
LOG_INTERVAL="${LOG_INTERVAL:-100}"
THRESHOLD_STEP="${THRESHOLD_STEP:-0.05}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="training_runs/paper_repro_${TIMESTAMP}"
mkdir -p "${RUN_ROOT}"

run_model_pipeline() {
  local gpu_id="$1"
  local preset="$2"
  local pretrain_dir="${RUN_ROOT}/pretrain_${preset}"
  local finetune_dir="${RUN_ROOT}/finetune_${preset}"

  CUDA_VISIBLE_DEVICES="${gpu_id}" python3 -u repro/train.py \
    --phase pretrain \
    --datasets-dir datasets \
    --preset "${preset}" \
    --epochs "${PRETRAIN_EPOCHS}" \
    --batch-size 1 \
    --device cuda:0 \
    --output-dir "${pretrain_dir}" \
    --run-name main \
    --num-workers "${NUM_WORKERS}" \
    --threshold-step "${THRESHOLD_STEP}" \
    --log-interval "${LOG_INTERVAL}" \
    --positive-weight 1.0 \
    --standardize-input

  CUDA_VISIBLE_DEVICES="${gpu_id}" python3 -u repro/train.py \
    --phase finetune \
    --datasets-dir datasets \
    --preset "${preset}" \
    --epochs "${FINETUNE_EPOCHS}" \
    --batch-size 1 \
    --device cuda:0 \
    --output-dir "${finetune_dir}" \
    --run-name main \
    --num-workers "${NUM_WORKERS}" \
    --threshold-step "${THRESHOLD_STEP}" \
    --log-interval "${LOG_INTERVAL}" \
    --positive-weight 1.0 \
    --standardize-input \
    --init-checkpoint "${pretrain_dir}/main/best.pt"
}

(
  run_model_pipeline "${GPU_A}" model0
  run_model_pipeline "${GPU_A}" model2
  run_model_pipeline "${GPU_A}" model4
) &
PID_A=$!

(
  run_model_pipeline "${GPU_B}" model1
  run_model_pipeline "${GPU_B}" model3
) &
PID_B=$!

wait "${PID_A}" "${PID_B}"

python3 -u repro/eval.py \
  --phase finetune \
  --datasets-dir datasets \
  --device cpu \
  --batch-size 1 \
  --num-workers 0 \
  --output-json "${RUN_ROOT}/ensemble_summary.json" \
  --checkpoints \
  "${RUN_ROOT}/finetune_model0/main/best.pt" \
  "${RUN_ROOT}/finetune_model1/main/best.pt" \
  "${RUN_ROOT}/finetune_model2/main/best.pt" \
  "${RUN_ROOT}/finetune_model3/main/best.pt" \
  "${RUN_ROOT}/finetune_model4/main/best.pt"

printf '%s\n' "run_root=${RUN_ROOT}"
