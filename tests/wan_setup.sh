#!/usr/bin/env bash
# ===== WAN 性能测试一键脚本 (A=LAN, B=公网) =====
# 在局域网机器A上运行, 自动通过SSH控制公网机器B。
#
# 用法:
#   bash tests/wan_setup.sh
#
# 前置: 机器B sshd_config 中 GatewayPorts yes
# 前置: 机器B 上已安装 conda env, Python 环境与 A 一致
# 前置: A -> B 已配置 SSH 公钥免密登录
#
# 修改下方配置区即可.
set -euo pipefail

# =========================== 配置区 ===========================
B_PUBLIC_IP="1.2.3.4"                         # 机器B 的公网 IP
B_SSH_USER="ubuntu"                            # 机器B 的 SSH 用户名 (需 passwordless sudo)
B_SSH_KEY="~/.ssh/id_rsa"                      # SSH 私钥路径

CONDA_ENV="sf"                                 # conda 环境名
A_TC_DEVICE="eth0"                             # A 侧受 tc 限制的网卡
B_TC_DEVICE="eth0"                             # B 侧受 tc 限制的网卡

RAY_PORT=20001                                 # Ray Head 端口
RAY_OBJECT_STORE_MEMORY=2000000000             # object store 内存 (2 GB)
RAY_NUM_CPUS=16                                # 每节点 CPU 数

A_COMPANY_SPU_PORT=9394
A_COORDINATOR_SPU_PORT=9396
A_NODE_MANAGER_PORT=20003
A_OBJECT_MANAGER_PORT=20002
A_MIN_WORKER_PORT=10060
A_MAX_WORKER_PORT=10160

B_PARTNER_SPU_PORT=9395
B_NODE_MANAGER_PORT=54001
B_OBJECT_MANAGER_PORT=54002
B_MIN_WORKER_PORT=54003
B_MAX_WORKER_PORT=54103

# 测试网络条件: "带宽MB/s:延迟ms" 逗号分隔
WAN_CONDITIONS="10:20,10:50,25:20,25:50,50:20,50:50"

PYTEST_TARGET="tests/test_performance.py::TestWANNetworkImpact"
# ==============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ------------------------------------------------------------------
# 自动检测 A 的局域网 IP
# ------------------------------------------------------------------
A_IP=$(ip -4 route get 8.8.8.8 2>/dev/null | awk '{print $7; exit}' || true)
if [ -z "${A_IP}" ]; then
    A_IP=$(hostname -I 2>/dev/null | awk '{print $1}') || true
fi
if [ -z "${A_IP}" ]; then
    echo "无法自动检测本机 IP, 请在配置区手动设置 A_IP."
    exit 1
fi
echo "检测到 A 的局域网 IP: ${A_IP}"

# ------------------------------------------------------------------
# 快捷 SSH 到 B (使用指定公钥)
# ------------------------------------------------------------------
_ssh_b() { ssh -i "${B_SSH_KEY}" "${B_SSH_USER}@${B_PUBLIC_IP}" "$@"; }

# ------------------------------------------------------------------
# 导出配置给 test_performance.py
# ------------------------------------------------------------------
export PERF_RUN_WAN=1
export PERF_USE_ENV_CONFIG=1
export PERF_A_IP="${A_IP}"
export PERF_A_RAY_PORT="${RAY_PORT}"
export PERF_A_OBJECT_MANAGER_PORT="${A_OBJECT_MANAGER_PORT}"
export PERF_A_NODE_MANAGER_PORT="${A_NODE_MANAGER_PORT}"
export PERF_A_MIN_WORKER_PORT="${A_MIN_WORKER_PORT}"
export PERF_A_MAX_WORKER_PORT="${A_MAX_WORKER_PORT}"
export PERF_A_COMPANY_SPU_PORT="${A_COMPANY_SPU_PORT}"
export PERF_A_COORDINATOR_SPU_PORT="${A_COORDINATOR_SPU_PORT}"
export PERF_B_IP="127.0.0.1"
export PERF_B_PARTNER_SPU_PORT="${B_PARTNER_SPU_PORT}"
export PERF_RAY_NUM_CPUS="${RAY_NUM_CPUS}"
export PERF_RAY_OBJECT_STORE_MEMORY="${RAY_OBJECT_STORE_MEMORY}"

export PERF_TC_DEVICE="${A_TC_DEVICE}"
export PERF_WAN_TC_REMOTE_HOST="${B_PUBLIC_IP}"
export PERF_WAN_TC_REMOTE_PORT="22"
export PERF_WAN_TC_REMOTE_USER="${B_SSH_USER}"
export PERF_WAN_TC_REMOTE_KEY="${B_SSH_KEY}"
export PERF_WAN_TC_REMOTE_DEVICE="${B_TC_DEVICE}"
export PERF_WAN_CONDITIONS="${WAN_CONDITIONS}"

# ------------------------------------------------------------------
# 清理 (A 本地 + B 远程)
# ------------------------------------------------------------------
cleanup() {
    echo "=== 清理 ==="
    kill %1 %2 2>/dev/null || true
    sudo tc qdisc del dev "${A_TC_DEVICE}" root 2>/dev/null || true
    _ssh_b "sudo tc qdisc del dev ${B_TC_DEVICE} root" 2>/dev/null || true
    _ssh_b "ray stop --force" 2>/dev/null || true
    ray stop --force 2>/dev/null || true
}
trap cleanup EXIT

# ------------------------------------------------------------------
# Step 1: 建立 A -> B 的反向隧道
# ------------------------------------------------------------------
echo "[1/6] 建立 SSH 反向隧道 (A -> B)..."

SSH_FORWARDS=(
    -N -T
    -o ServerAliveInterval=30 -o ServerAliveCountMax=3
    -o ExitOnForwardFailure=yes
    -R "0.0.0.0:${RAY_PORT}:localhost:${RAY_PORT}"
    -R "0.0.0.0:${A_NODE_MANAGER_PORT}:localhost:${A_NODE_MANAGER_PORT}"
    -R "0.0.0.0:${A_OBJECT_MANAGER_PORT}:localhost:${A_OBJECT_MANAGER_PORT}"
    -R "0.0.0.0:${A_COMPANY_SPU_PORT}:localhost:${A_COMPANY_SPU_PORT}"
    -R "0.0.0.0:${A_COORDINATOR_SPU_PORT}:localhost:${A_COORDINATOR_SPU_PORT}"
)
for p in $(seq "${A_MIN_WORKER_PORT}" "${A_MAX_WORKER_PORT}"); do
    SSH_FORWARDS+=(-R "0.0.0.0:${p}:localhost:${p}")
done

if command -v autossh &>/dev/null; then
    autossh -M 0 -i "${B_SSH_KEY}" "${SSH_FORWARDS[@]}" "${B_SSH_USER}@${B_PUBLIC_IP}" &
else
    ssh -i "${B_SSH_KEY}" "${SSH_FORWARDS[@]}" "${B_SSH_USER}@${B_PUBLIC_IP}" &
fi
sleep 3

# ------------------------------------------------------------------
# Step 2: A 启动 Ray Head
# ------------------------------------------------------------------
echo "[2/6] A 启动 Ray Head..."

export RAY_memory_usage_threshold="${RAY_memory_usage_threshold:-0.99}"
export RAY_memory_monitor_refresh_ms="${RAY_memory_monitor_refresh_ms:-0}"

ray stop --force 2>/dev/null || true

ray start --head \
    --port="${RAY_PORT}" \
    --node-ip-address="${A_IP}" \
    --num-cpus="${RAY_NUM_CPUS}" \
    --object-store-memory="${RAY_OBJECT_STORE_MEMORY}" \
    --resources='{"company": 10, "coordinator": 10}' \
    --node-manager-port="${A_NODE_MANAGER_PORT}" \
    --object-manager-port="${A_OBJECT_MANAGER_PORT}" \
    --min-worker-port="${A_MIN_WORKER_PORT}" \
    --max-worker-port="${A_MAX_WORKER_PORT}"

# ------------------------------------------------------------------
# Step 3: B 启动 Partner Worker
# ------------------------------------------------------------------
echo "[3/6] B 启动 Partner Worker..."

_ssh_b "ray stop --force" 2>/dev/null || true
_ssh_b "
    export RAY_memory_usage_threshold=0.99
    export RAY_memory_monitor_refresh_ms=0
    ray start \
        --address='localhost:${RAY_PORT}' \
        --node-ip-address='127.0.0.1' \
        --num-cpus='${RAY_NUM_CPUS}' \
        --object-store-memory='${RAY_OBJECT_STORE_MEMORY}' \
        --resources='{\"partner\": 10}' \
        --node-manager-port='${B_NODE_MANAGER_PORT}' \
        --object-manager-port='${B_OBJECT_MANAGER_PORT}' \
        --min-worker-port='${B_MIN_WORKER_PORT}' \
        --max-worker-port='${B_MAX_WORKER_PORT}'
"

# ------------------------------------------------------------------
# Step 4: 等待 Partner Worker 上线
# ------------------------------------------------------------------
echo "[4/6] 等待 Partner Worker 上线..."
for i in $(seq 1 30); do
    if ray status --address="localhost:${RAY_PORT}" 2>/dev/null | grep -q partner; then
        echo "Partner Worker 在线."
        break
    fi
    sleep 2
done
if ! ray status --address="localhost:${RAY_PORT}" 2>/dev/null | grep -q partner; then
    echo "Partner Worker 未能上线, 终止."
    exit 1
fi

# ------------------------------------------------------------------
# Step 5: 运行 WAN 性能测试
# ------------------------------------------------------------------
echo "[5/6] 运行 WAN 性能测试..."
echo "======================================"
echo " A (LAN):          ${A_IP}"
echo " B (公网):         ${B_PUBLIC_IP}"
echo " A tc device:      ${A_TC_DEVICE}"
echo " B tc device:      ${B_TC_DEVICE}"
echo " WAN conditions:   ${WAN_CONDITIONS}"
echo " Conda env:        ${CONDA_ENV}"
echo "======================================"

cd "${PROJECT_ROOT}"

PYTEST_TIMEOUT_ARGS=()
if conda run -n "${CONDA_ENV}" pytest --help 2>/dev/null | grep -q -- "--timeout"; then
    PYTEST_TIMEOUT_ARGS=(--timeout=0)
fi

conda run -n "${CONDA_ENV}" pytest "${PYTEST_TARGET}" \
    -m "performance and wan and not slow" \
    "${PYTEST_TIMEOUT_ARGS[@]}" \
    -rs "$@"

echo ""
echo "[6/6] WAN 性能测试完成。"
echo "结果: test_results/performance/psi_network_impact_wan.csv"
echo "      test_results/performance/sslr_network_impact_wan.csv"
echo "      test_results/performance/xgboost_network_impact_wan.csv"
