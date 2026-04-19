#!/usr/bin/env bash
set -euo pipefail

# === 运行推理任务（假设Ray已经启动）===
A_IP="210.28.133.104"
B_IP="210.28.133.104"
PORT_RAY="20001"
PORT_COMPANY_SPU="11001"
PORT_PARTNER_SPU="11002"
PORT_COORD_SPU="11003"

cd "$(dirname "$0")"
COM_PATH="."
PAR_PATH="../partner"

MODEL="${1:-SSLR}"
COMPANY_DATA_PATH="${2:-${COM_PATH}/host_test.csv}"
PARTNER_DATA_PATH="${3:-${PAR_PATH}/guest_test.csv}"

if [ "${MODEL}" = "SSXGBoost" ]; then
  COMPANY_MODEL_DIR="${COM_PATH}/models/xgb_company"
  PARTNER_MODEL_DIR="${PAR_PATH}/models/xgb_partner"
else
  COMPANY_MODEL_DIR="${COM_PATH}/models/lr_company"
  PARTNER_MODEL_DIR="${PAR_PATH}/models/lr_partner"
fi

if [ ! -d "${COMPANY_MODEL_DIR}" ]; then
  echo "Company 模型目录不存在: ${COMPANY_MODEL_DIR}" >&2
  exit 1
fi

if [ ! -d "${PARTNER_MODEL_DIR}" ]; then
  echo "Partner 模型目录不存在: ${PARTNER_MODEL_DIR}" >&2
  exit 1
fi

echo "正在启动推理任务 (Model=${MODEL})..."

# 运行推理
python3 infer_run.py \
  --mode="multi_distributed" \
  --ray_head_addr="${A_IP}:${PORT_RAY}" \
  --company_spu_addr="${A_IP}:${PORT_COMPANY_SPU}" \
  --partner_spu_addr="${B_IP}:${PORT_PARTNER_SPU}" \
  --coordinator_spu_addr="${A_IP}:${PORT_COORD_SPU}" \
  --model="${MODEL}" \
  --company_model_path="${COMPANY_MODEL_DIR}" \
  --partner_model_path="${PARTNER_MODEL_DIR}" \
  --company_data_path="${COMPANY_DATA_PATH}" \
  --partner_data_path="${PARTNER_DATA_PATH}" \
  2> >(grep -v -E "(openssl_factory|Yacl has been configured|entropy.*source)" >&2)