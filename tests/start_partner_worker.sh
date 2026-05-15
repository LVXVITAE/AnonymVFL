#!/usr/bin/env bash
# ===== Partner Worker 启动脚本 =====
# 在 Machine B (远程机器) 上运行此脚本, 加入 Machine A 的 Ray 集群
#
# 用法:
#   1. 确保 Machine B 上已安装相同的 Python 环境和依赖
#   2. 修改下方的 RAY_HEAD_ADDR 为 Machine A 的实际地址
#   3. 运行: bash tests/start_partner_worker.sh
#
# 或者直接传参:
#   bash tests/start_partner_worker.sh <RAY_HEAD_IP> <RAY_HEAD_PORT>

set -e

# ---------- 从 distributed_config.yaml 自动解析配置 ----------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG="${SCRIPT_DIR}/distributed_config.yaml"

if [ ! -f "${CONFIG}" ]; then
    echo "❌ 找不到配置文件: ${CONFIG}"
    exit 1
fi

RAY_HEAD_IP=$(python3 -c "import yaml; c=yaml.safe_load(open('${CONFIG}')); print(c['machine_a']['ip'])")
RAY_HEAD_PORT=$(python3 -c "import yaml; c=yaml.safe_load(open('${CONFIG}')); print(c['machine_a']['ray_port'])")
PARTNER_IP=$(python3 -c "import yaml; c=yaml.safe_load(open('${CONFIG}')); print(c['machine_b']['ip'])")
PARTNER_SPU_PORT=$(python3 -c "import yaml; c=yaml.safe_load(open('${CONFIG}')); print(c['machine_b']['partner_spu_port'])")
NUM_CPUS=$(python3 -c "import yaml; c=yaml.safe_load(open('${CONFIG}')); print(c['ray']['num_cpus'])")
OBJECT_STORE_MEMORY=$(python3 -c "import yaml; c=yaml.safe_load(open('${CONFIG}')); print(c['ray']['object_store_memory'])")

# 允许命令行参数覆盖
RAY_HEAD_IP="${1:-${RAY_HEAD_IP}}"
RAY_HEAD_PORT="${2:-${RAY_HEAD_PORT}}"
PARTNER_IP="${3:-${PARTNER_IP}}"

RAY_HEAD_ADDR="${RAY_HEAD_IP}:${RAY_HEAD_PORT}"
# ---------- 配置区结束 ----------

echo "======================================"
echo " AnonymVFL Partner Worker 启动脚本"
echo "======================================"

# 默认放宽 Ray OOM 保护，避免分布式任务中 worker 被内存阈值提前杀掉。
# 允许用户通过环境变量覆盖。
export RAY_memory_usage_threshold="${RAY_memory_usage_threshold:-0.99}"
export RAY_memory_monitor_refresh_ms="${RAY_memory_monitor_refresh_ms:-0}"

echo " Ray Head 地址: ${RAY_HEAD_ADDR}"
echo " Partner 本机 IP: ${PARTNER_IP}"
echo " Partner SPU 端口: ${PARTNER_SPU_PORT}"
echo " CPU 数量: ${NUM_CPUS}"
echo " Mem Threshold: ${RAY_memory_usage_threshold}"
echo " Mem Monitor(ms): ${RAY_memory_monitor_refresh_ms}"
echo "======================================"

if [ "${PARTNER_IP}" = "${RAY_HEAD_IP}" ]; then
    echo "检测到 Machine A 与 Machine B 使用同一 IP (${PARTNER_IP})。"
    echo "这是单机退化模式: partner 资源应由 Ray head 节点直接声明。"
    echo "请先运行: bash tests/start_ray_head.sh"
    echo "如果 head 已启动且 ray status 中包含 partner 资源, 无需再启动 Partner Worker。"
    ray status || true
    exit 0
fi

# 1. 停止可能残留的 Ray 进程
ray stop --force 2>/dev/null || true

# 2. 以 Worker 身份加入 Ray 集群
echo "[1/2] 加入 Ray 集群..."
ray start \
    --address="${RAY_HEAD_ADDR}" \
    --node-ip-address="${PARTNER_IP}" \
    --num-cpus="${NUM_CPUS}" \
    --object-store-memory="${OBJECT_STORE_MEMORY}" \
    --resources='{"partner": 10}' \
    --node-manager-port=54001 \
    --object-manager-port=54002 \
    --min-worker-port=54003 \
    --max-worker-port=54103

echo "[2/2] Partner Worker 已加入集群, 保持运行中..."
echo "按 Ctrl+C 退出"

# 3. 保持进程运行 (等待任务分配)
tail -f /dev/null
