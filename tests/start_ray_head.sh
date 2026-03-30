#!/usr/bin/env bash
set -euo pipefail

# ===== Ray Head 启动脚本 (Machine A) =====
# 从 distributed_config.yaml 读取配置, 启动 Ray head 并分配 company + coordinator 资源
#
# 用法:
#   bash tests/start_ray_head.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG="${SCRIPT_DIR}/distributed_config.yaml"

if [ ! -f "${CONFIG}" ]; then
    echo "❌ 找不到配置文件: ${CONFIG}"
    exit 1
fi

# 从 YAML 中解析配置 (兼容无 yq 的环境)
A_IP=$(python3 -c "import yaml; c=yaml.safe_load(open('${CONFIG}')); print(c['machine_a']['ip'])")
RAY_PORT=$(python3 -c "import yaml; c=yaml.safe_load(open('${CONFIG}')); print(c['machine_a']['ray_port'])")
OBJ_MGR_PORT=$(python3 -c "import yaml; c=yaml.safe_load(open('${CONFIG}')); print(c['machine_a'].get('object_manager_port', 0))")
NODE_MGR_PORT=$(python3 -c "import yaml; c=yaml.safe_load(open('${CONFIG}')); print(c['machine_a'].get('node_manager_port', 0))")
MIN_WORKER_PORT=$(python3 -c "import yaml; c=yaml.safe_load(open('${CONFIG}')); print(c['machine_a'].get('min_worker_port', 0))")
MAX_WORKER_PORT=$(python3 -c "import yaml; c=yaml.safe_load(open('${CONFIG}')); print(c['machine_a'].get('max_worker_port', 0))")
NUM_CPUS=$(python3 -c "import yaml; c=yaml.safe_load(open('${CONFIG}')); print(c['ray']['num_cpus'])")
OBJ_STORE=$(python3 -c "import yaml; c=yaml.safe_load(open('${CONFIG}')); print(c['ray']['object_store_memory'])")
COMPANY_RES=$(python3 -c "import yaml; c=yaml.safe_load(open('${CONFIG}')); print(c['machine_a']['ray_resources']['company'])")
COORD_RES=$(python3 -c "import yaml; c=yaml.safe_load(open('${CONFIG}')); print(c['machine_a']['ray_resources']['coordinator'])")

echo "======================================"
echo " AnonymVFL Ray Head 启动脚本"
echo "======================================"
echo " Node IP:          ${A_IP}"
echo " Ray Port:         ${RAY_PORT}"
echo " Object Mgr Port:  ${OBJ_MGR_PORT}"
echo " Node Mgr Port:    ${NODE_MGR_PORT}"
echo " Worker Ports:     ${MIN_WORKER_PORT}-${MAX_WORKER_PORT}"
echo " CPUs:             ${NUM_CPUS}"
echo " Object Store:     ${OBJ_STORE}"
echo " Resources:        company=${COMPANY_RES}, coordinator=${COORD_RES}"
echo "======================================"

# 停止可能残留的 Ray 进程
ray stop --force 2>/dev/null || true
sleep 1

# 构建可选端口参数
PORT_ARGS=""
if [ "${OBJ_MGR_PORT}" != "0" ]; then
    PORT_ARGS="${PORT_ARGS} --object-manager-port=${OBJ_MGR_PORT}"
fi
if [ "${NODE_MGR_PORT}" != "0" ]; then
    PORT_ARGS="${PORT_ARGS} --node-manager-port=${NODE_MGR_PORT}"
fi
if [ "${MIN_WORKER_PORT}" != "0" ]; then
    PORT_ARGS="${PORT_ARGS} --min-worker-port=${MIN_WORKER_PORT}"
fi
if [ "${MAX_WORKER_PORT}" != "0" ]; then
    PORT_ARGS="${PORT_ARGS} --max-worker-port=${MAX_WORKER_PORT}"
fi

# 启动 Ray head
echo "[1/3] 启动 Ray head..."
ray start --head \
    --node-ip-address="${A_IP}" \
    --port="${RAY_PORT}" \
    --num-cpus="${NUM_CPUS}" \
    --resources="{\"company\": ${COMPANY_RES}, \"coordinator\": ${COORD_RES}}" \
    --object-store-memory="${OBJ_STORE}" \
    --system-config='{"max_direct_call_object_size": 104857600, "task_rpc_inlined_bytes_limit": 104857600}' \
    ${PORT_ARGS}

echo "[2/3] 等待 GCS 服务就绪..."
MAX_RETRIES=10
RETRY_COUNT=0
while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if ray status >/dev/null 2>&1; then
        echo "✅ Ray head 启动成功, GCS 已就绪"
        break
    fi
    RETRY_COUNT=$((RETRY_COUNT + 1))
    echo "  等待中... (${RETRY_COUNT}/${MAX_RETRIES})"
    sleep 2
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo "❌ GCS 服务启动超时"
    exit 1
fi

# 显示集群状态
echo "[3/3] 当前集群状态:"
ray status

echo ""
echo "📍 Ray 集群地址: ${A_IP}:${RAY_PORT}"
echo "💡 在 Machine B 上运行: bash tests/start_partner_worker.sh ${A_IP} ${RAY_PORT}"
echo "💡 停止集群: ray stop"
