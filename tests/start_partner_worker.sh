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

# ---------- 配置区 ----------
RAY_HEAD_IP="${1:-MACHINE_A_IP}"       # ← 替换或通过参数传入 Machine A 的 IP
RAY_HEAD_PORT="${2:-20001}"            # ← Ray head 端口
PARTNER_SPU_PORT="${3:-9395}"          # ← Partner SPU 监听端口

NUM_CPUS="${4:-8}"
OBJECT_STORE_MEMORY="${5:-2000000000}" # 2 GB

RAY_HEAD_ADDR="${RAY_HEAD_IP}:${RAY_HEAD_PORT}"
# ---------- 配置区结束 ----------

echo "======================================"
echo " AnonymVFL Partner Worker 启动脚本"
echo "======================================"
echo " Ray Head 地址: ${RAY_HEAD_ADDR}"
echo " Partner SPU 端口: ${PARTNER_SPU_PORT}"
echo " CPU 数量: ${NUM_CPUS}"
echo "======================================"

# 1. 停止可能残留的 Ray 进程
ray stop --force 2>/dev/null || true

# 2. 以 Worker 身份加入 Ray 集群
echo "[1/2] 加入 Ray 集群..."
ray start \
    --address="${RAY_HEAD_ADDR}" \
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
