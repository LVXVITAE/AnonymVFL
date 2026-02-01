#!/usr/bin/env bash
set -euo pipefail

# === 运行训练任务（假设Ray已经启动）===
A_IP="210.28.133.104"
B_IP="210.28.133.104"
PORT_RAY="20001"
PORT_COMPANY_SPU="11001"
PORT_PARTNER_SPU="11002"
PORT_COORD_SPU="11003"

cd "$(dirname "$0")"
COM_PATH="."
PAR_PATH="../partner"

N_EPOCHS="${1:-10}"
BATCH_SIZE="${2:-1000}"
LEARNING_RATE="${3:-0.1}"
VAL_STEPS="${4:-1}"
MODEL="${5:-SSLR}"
N_ESTIMATORS="${6:-2}"
MAX_DEPTH="${7:-2}"
K_QUANTILES="${8:-1}"
REG_COEF="${9:-0.0}"
BUCKETS_PATH="${10:-${COM_PATH}/buckets.npy}"

if [ "${MODEL}" = "SSXGBoost" ]; then
  COMPANY_MODEL_DIR="${COM_PATH}/models/xgb_company"
  PARTNER_MODEL_DIR="${PAR_PATH}/models/xgb_partner"
else
  COMPANY_MODEL_DIR="${COM_PATH}/models/lr_company"
  PARTNER_MODEL_DIR="${PAR_PATH}/models/lr_partner"
fi

mkdir -p "${COMPANY_MODEL_DIR}" "${PARTNER_MODEL_DIR}"

echo "正在启动训练任务 (Model=${MODEL})..."

# 运行训练
python3 truerun.py \
  --mode="multi_distributed" \
  --ray_head_addr="${A_IP}:${PORT_RAY}" \
  --company_spu_addr="${A_IP}:${PORT_COMPANY_SPU}" \
  --partner_spu_addr="${B_IP}:${PORT_PARTNER_SPU}" \
  --coordinator_spu_addr="${A_IP}:${PORT_COORD_SPU}" \
  --run_psi=True \
  \
  --path_to_company_train_dataset="${COM_PATH}/host_train.csv" \
  --path_to_company_val_dataset="${COM_PATH}/host_test.csv" \
  --path_to_company_share="${COM_PATH}/company_share.csv" \
  --path_to_company_model_save_dir="${COMPANY_MODEL_DIR}" \
  --share_y=False \
  \
  --path_to_partner_train_dataset="${PAR_PATH}/guest_train.csv" \
  --path_to_partner_val_dataset="${PAR_PATH}/guest_test.csv" \
  --path_to_partner_share="${PAR_PATH}/partner_share.csv" \
  --path_to_partner_model_save_dir="${PARTNER_MODEL_DIR}" \
  --path_to_buckets="${BUCKETS_PATH}" \
  \
  --model="${MODEL}" \
  --n_epochs=${N_EPOCHS} \
  --batch_size=${BATCH_SIZE} \
  --val_steps=${VAL_STEPS} \
  --lr=${LEARNING_RATE} \
  --n_estimators=${N_ESTIMATORS} \
  --max_depth=${MAX_DEPTH} \
  --K_quantiles=${K_QUANTILES} \
  --reg_coef=${REG_COEF} \
  2> >(grep -v -E "(openssl_factory|Yacl has been configured|entropy.*source)" >&2)

