#!/usr/bin/env bash
set -euo pipefail

# === 运行推理任务（假设 Ray 已启动，地址由环境变量控制）===
RAY_HEAD_ADDR="${RAY_HEAD_ADDR:-210.28.133.104:20001}"
COMPANY_SPU_ADDR="${COMPANY_SPU_ADDR:-210.28.133.104:11001}"
PARTNER_SPU_ADDR="${PARTNER_SPU_ADDR:-210.28.133.105:11002}"
COORDINATOR_SPU_ADDR="${COORDINATOR_SPU_ADDR:-210.28.133.106:11003}"

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
echo "Ray Head: ${RAY_HEAD_ADDR}"
echo "Company SPU: ${COMPANY_SPU_ADDR}"
echo "Partner SPU: ${PARTNER_SPU_ADDR}"
echo "Coordinator SPU: ${COORDINATOR_SPU_ADDR}"

# 运行推理
python3 infer_run.py \
  --mode="multi_distributed" \
  --ray_head_addr="${RAY_HEAD_ADDR}" \
  --company_spu_addr="${COMPANY_SPU_ADDR}" \
  --partner_spu_addr="${PARTNER_SPU_ADDR}" \
  --coordinator_spu_addr="${COORDINATOR_SPU_ADDR}" \
  --model="${MODEL}" \
  --company_model_path="${COMPANY_MODEL_DIR}" \
  --partner_model_path="${PARTNER_MODEL_DIR}" \
  --company_data_path="${COMPANY_DATA_PATH}" \
  --partner_data_path="${PARTNER_DATA_PATH}" \
  2> >(grep -v -E "(openssl_factory|Yacl has been configured|entropy.*source)" >&2)