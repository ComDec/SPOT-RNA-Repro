#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:-1}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-1}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-3}"
THRESHOLD_STEP="${THRESHOLD_STEP:-0.1}"
LOG_INTERVAL="${LOG_INTERVAL:-200}"
NUM_WORKERS="${NUM_WORKERS:-2}"
PRESETS="${PRESETS:-model0 model1 model2 model3 model4}"
DROP_MULTIPLETS="${DROP_MULTIPLETS:-1}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="training_runs/repro_${TIMESTAMP}"
mkdir -p "${RUN_ROOT}"

DROP_ARGS=()
if [ "${DROP_MULTIPLETS}" = "1" ]; then
  DROP_ARGS+=(--drop-multiplets)
fi

BEST_CHECKPOINTS=()

for PRESET in ${PRESETS}; do
  PRETRAIN_RUN="${RUN_ROOT}/pretrain_${PRESET}"
  FINETUNE_RUN="${RUN_ROOT}/finetune_${PRESET}"

  CUDA_VISIBLE_DEVICES="${GPU_ID}" python3 -u train_spotrna.py \
    --phase pretrain \
    --datasets-dir datasets \
    --preset "${PRESET}" \
    --epochs "${PRETRAIN_EPOCHS}" \
    --batch-size 1 \
    --device cuda:0 \
    --output-dir "${PRETRAIN_RUN}" \
    --run-name main \
    --num-workers "${NUM_WORKERS}" \
    --threshold-step "${THRESHOLD_STEP}" \
    --log-interval "${LOG_INTERVAL}" \
    "${DROP_ARGS[@]}"

  CUDA_VISIBLE_DEVICES="${GPU_ID}" python3 -u train_spotrna.py \
    --phase finetune \
    --datasets-dir datasets \
    --preset "${PRESET}" \
    --epochs "${FINETUNE_EPOCHS}" \
    --batch-size 1 \
    --device cuda:0 \
    --output-dir "${FINETUNE_RUN}" \
    --run-name main \
    --num-workers "${NUM_WORKERS}" \
    --threshold-step "${THRESHOLD_STEP}" \
    --log-interval "${LOG_INTERVAL}" \
    --init-checkpoint "${PRETRAIN_RUN}/main/best.pt" \
    "${DROP_ARGS[@]}"

  BEST_CHECKPOINTS+=("${FINETUNE_RUN}/main/best.pt")
done

CUDA_VISIBLE_DEVICES="${GPU_ID}" python3 -u evaluate_spotrna_ensemble.py \
  --phase finetune \
  --datasets-dir datasets \
  --device cuda:0 \
  --batch-size 1 \
  --num-workers "${NUM_WORKERS}" \
  --output-json "${RUN_ROOT}/ensemble_summary.json" \
  "${DROP_ARGS[@]}" \
  --checkpoints "${BEST_CHECKPOINTS[@]}"

printf '%s\n' "run_root=${RUN_ROOT}"
