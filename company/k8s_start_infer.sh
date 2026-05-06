#!/usr/bin/env bash
set -euo pipefail

# K8s 部署模式推理脚本（假设 Ray 已由 Deployment 自动启动）

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
  echo "错误: Company 模型目录不存在: ${COMPANY_MODEL_DIR}" >&2
  exit 1
fi

if [ ! -d "${PARTNER_MODEL_DIR}" ]; then
  echo "错误: Partner 模型目录不存在: ${PARTNER_MODEL_DIR}" >&2
  exit 1
fi

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

echo "=== K8s 推理模式 ==="
echo "Company IP:       ${A_IP}"
echo "Partner SPU Addr: ${B_ADDR}"
echo "Ray Head:         ${A_IP}:${PORT_RAY}"
echo "Coordinator SPU:  ${COORDINATOR_SPU_ADDR:-${A_IP}:${PORT_COORD_SPU}}"
echo "Model:            ${MODEL}"

if ! ray status &>/dev/null; then
  echo "警告: ray status 检查失败，将继续尝试执行推理" >&2
fi

python3 infer_run.py \
  --mode="multi_distributed" \
  --ray_head_addr="${A_IP}:${PORT_RAY}" \
  --company_spu_addr="${A_IP}:${PORT_COMPANY_SPU}" \
  --partner_spu_addr="${B_ADDR}" \
  --coordinator_spu_addr="${COORDINATOR_SPU_ADDR:-${A_IP}:${PORT_COORD_SPU}}" \
  --model="${MODEL}" \
  --company_model_path="${COMPANY_MODEL_DIR}" \
  --partner_model_path="${PARTNER_MODEL_DIR}" \
  --company_data_path="${COMPANY_DATA_PATH}" \
  --partner_data_path="${PARTNER_DATA_PATH}" \
  2> >(grep -v -E "(openssl_factory|Yacl has been configured|entropy.*source)" >&2)
