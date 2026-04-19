#!/usr/bin/env bash
set -euo pipefail

# K8s 部署模式训练脚本（假设 Ray 已由 Deployment 自动启动）

A_IP="${POD_IP:-${HOST_IP:-}}"
if [ -z "${A_IP}" ]; then
  echo "错误: 未检测到 POD_IP/HOST_IP，请在 Kubernetes Company 容器内运行" >&2
  exit 1
fi

PORT_RAY="${RAY_PORT:-6379}"
PORT_COMPANY_SPU="${COMPANY_SPU_PORT:-9394}"
PORT_PARTNER_SPU="${PARTNER_SPU_PORT:-9395}"
PORT_COORD_SPU="${COORD_SPU_PORT:-9396}"

cd "$(dirname "$0")"
COM_PATH="."
PAR_PATH="../partner"

# 训练参数（与 start_training.sh 保持一致）
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

# Partner 地址解析：优先使用环境变量（跨集群），否则尝试解析 service。
if [ -n "${PARTNER_SPU_ADDR:-}" ]; then
  B_ADDR="${PARTNER_SPU_ADDR}"
else
  RELEASE_NAME=""
  if [ -n "${POD_NAME:-}" ]; then
    RELEASE_NAME="$(echo "${POD_NAME}" | sed 's/-mobile-mpc-project-company.*//')"
  fi

  if [ -n "${RELEASE_NAME}" ]; then
    PARTNER_SVC="${RELEASE_NAME}-mobile-mpc-project-partner-svc"
    B_IP="$(getent hosts "${PARTNER_SVC}" 2>/dev/null | awk '{print $1}' | head -1)"
  else
    B_IP=""
  fi

  if [ -z "${B_IP}" ]; then
    echo "错误: 无法解析 Partner 地址。请设置 PARTNER_SPU_ADDR，例如 10.0.0.2:${PORT_PARTNER_SPU}" >&2
    exit 1
  fi
  B_ADDR="${B_IP}:${PORT_PARTNER_SPU}"
fi

echo "=== K8s 训练模式 ==="
echo "Company IP:       ${A_IP}"
echo "Partner SPU Addr: ${B_ADDR}"
echo "Ray Head:         ${A_IP}:${PORT_RAY}"
echo "Model:            ${MODEL}"

if ! ray status &>/dev/null; then
  echo "警告: ray status 检查失败，将继续尝试执行训练" >&2
fi

python3 truerun.py \
  --mode="multi_distributed" \
  --ray_head_addr="${A_IP}:${PORT_RAY}" \
  --company_spu_addr="${A_IP}:${PORT_COMPANY_SPU}" \
  --partner_spu_addr="${B_ADDR}" \
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
