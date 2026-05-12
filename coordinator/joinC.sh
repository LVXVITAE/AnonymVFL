#!/usr/bin/env bash
set -euo pipefail

# === Coordinator 机器加入 Ray 集群 ===
# 说明：Coordinator 在裸机模式下作为独立 Ray Worker 加入集群，
# 训练/推理实际使用的 COORDINATOR_SPU_ADDR 由 Company 侧脚本传入。

RAY_HEAD_ADDR="${RAY_HEAD_ADDR:-210.28.133.104:20001}"
COORDINATOR_HOST="${COORDINATOR_HOST:-$(hostname -I 2>/dev/null | awk '{print $1}')}"
COORDINATOR_NUM_CPUS="${COORDINATOR_NUM_CPUS:-8}"
COORDINATOR_OBJECT_STORE_MEMORY="${COORDINATOR_OBJECT_STORE_MEMORY:-2000000000}"
COORDINATOR_NODE_MANAGER_PORT="${COORDINATOR_NODE_MANAGER_PORT:-55001}"
COORDINATOR_OBJECT_MANAGER_PORT="${COORDINATOR_OBJECT_MANAGER_PORT:-55002}"
COORDINATOR_MIN_WORKER_PORT="${COORDINATOR_MIN_WORKER_PORT:-55003}"
COORDINATOR_MAX_WORKER_PORT="${COORDINATOR_MAX_WORKER_PORT:-55103}"

cd "$(dirname "$0")"

if [ -z "${COORDINATOR_HOST}" ]; then
  echo "错误: 无法自动识别 COORDINATOR_HOST，请显式导出 Coordinator 机器 IP" >&2
  exit 1
fi

echo "正在将 Coordinator 机器加入 Ray 集群..."
echo "Ray Head:          ${RAY_HEAD_ADDR}"
echo "Coordinator Host:  ${COORDINATOR_HOST}"
echo "Coordinator SPU:   ${COORDINATOR_SPU_ADDR:-<由 Company 训练/推理脚本使用>}"

ray start --address "${RAY_HEAD_ADDR}" \
  --node-ip-address "${COORDINATOR_HOST}" \
  --node-manager-port="${COORDINATOR_NODE_MANAGER_PORT}" \
  --object-manager-port="${COORDINATOR_OBJECT_MANAGER_PORT}" \
  --min-worker-port="${COORDINATOR_MIN_WORKER_PORT}" \
  --max-worker-port="${COORDINATOR_MAX_WORKER_PORT}" \
  --dashboard-port=0 \
  --num-cpus "${COORDINATOR_NUM_CPUS}" \
  --resources='{"coordinator": 10}' \
  --object-store-memory="${COORDINATOR_OBJECT_STORE_MEMORY}"

echo "✅ Coordinator 已加入 Ray 集群"
