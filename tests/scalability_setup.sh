#!/usr/bin/env bash
# ===== 可扩展性/推理/批量测试一键脚本 =====
# 在机器A上运行, 自动控制机器B加入 Ray 集群, 然后执行所有非WAN非压力的性能测试。
#
# 支持两种模式:
#   双机模式: B_PUBLIC_IP != A_IP, 通过 SSH 控制机器B加入集群
#   单机退化模式: B_PUBLIC_IP == A_IP (自动检测), 在本机同时运行 Head + Worker
#
# 包含的测试:
#   TestPSIScalability       - PSI 不同样本量对齐时间
#   TestSSLRScalability      - SSLR 不同样本量训练时间
#   TestBatchSizeImpact      - batch_size 对训练时间的影响
#   TestSSXGBoostScalability - SSXGBoost 不同样本量训练时间
#   TestLRInferenceLatency   - SSLR 推理延迟
#   TestQuantileImpact       - k_quantiles 对 XGBoost 的影响
#   TestXGBoostInferenceLatency - XGBoost 推理延迟
#
# 用法:
#   bash tests/scalability_setup.sh                  # 自动检测双机/单机
#   B_PUBLIC_IP=自动检测到的A_IP bash ...             # 强制单机退化
#
# 前置 (双机): 机器B 上已安装 conda env sf, A 能通过 sshpass+密码 SSH 到 B
# 前置 (单机): 本机有足够内存/CPU 即可
#
# 修改下方配置区即可.
set -euo pipefail

# =========================== 配置区 ===========================
B_PUBLIC_IP="192.168.122.185"                  # 机器B 的 IP (若等于 A_IP 则自动进入单机退化模式)
B_SSH_USER="ubuntu"                            # 机器B 的 SSH 用户名
B_SSH_PASSWORD="ubuntu"                        # SSH 密码 (使用 sshpass)

CONDA_ENV="sf"                                 # conda 环境名

RAY_PORT=20001                                 # Ray Head 端口
RAY_OBJECT_STORE_MEMORY=4000000000             # object store 内存 (4 GB)
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

# 测试目标: 排除 WAN 和 Stress
PYTEST_TARGET="tests/test_performance.py"
# ==============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ------------------------------------------------------------------
# 自动检测 A 的局域网 IP 和网卡
# ------------------------------------------------------------------
_route_line=$(ip -4 route get 8.8.8.8 2>/dev/null || true)
A_IP=$(echo "${_route_line}" | sed -n 's/.*src \([0-9.]*\).*/\1/p' | head -1)
if [ -z "${A_IP}" ]; then
    A_IP=$(hostname -I 2>/dev/null | awk '{print $1}') || true
fi
if [ -z "${A_IP}" ]; then
    echo "无法自动检测本机 IP, 请在配置区手动设置 A_IP."
    exit 1
fi
A_NIC=$(echo "${_route_line}" | sed -n 's/.*dev \([^ ]*\).*/\1/p' | head -1)
echo "检测到 A 的局域网 IP: ${A_IP} (网卡: ${A_NIC:-未知})"

# ------------------------------------------------------------------
# 自动检测单机退化模式: B_IP == A_IP
# ------------------------------------------------------------------
if [ "${B_PUBLIC_IP}" = "${A_IP}" ]; then
    SINGLE_MACHINE=1
    B_EFFECTIVE_IP="${A_IP}"
    echo "*** 单机退化模式: B IP (${B_PUBLIC_IP}) == A IP (${A_IP}) ***"
else
    SINGLE_MACHINE=0
    B_EFFECTIVE_IP="${B_PUBLIC_IP}"
    echo "*** 双机模式: A=${A_IP} B=${B_PUBLIC_IP} ***"
fi

# ------------------------------------------------------------------
# 快捷 SSH 到 B (仅双机模式使用)
# ------------------------------------------------------------------
if [ "${SINGLE_MACHINE}" = "0" ]; then
    if ! command -v sshpass &>/dev/null; then
        echo "需要 sshpass, 正在安装: sudo apt-get install -y sshpass"
        sudo apt-get install -y sshpass
    fi
    _ssh_b() { sshpass -p "${B_SSH_PASSWORD}" ssh -o StrictHostKeyChecking=no "${B_SSH_USER}@${B_PUBLIC_IP}" "source ~/miniconda3/etc/profile.d/conda.sh && conda activate ${CONDA_ENV} && $*"; }
fi

# ------------------------------------------------------------------
# 导出配置给 test_performance.py
# ------------------------------------------------------------------
export PERF_USE_ENV_CONFIG=1
export PERF_A_IP="${A_IP}"
export PERF_A_RAY_PORT="${RAY_PORT}"
export PERF_A_OBJECT_MANAGER_PORT="${A_OBJECT_MANAGER_PORT}"
export PERF_A_NODE_MANAGER_PORT="${A_NODE_MANAGER_PORT}"
export PERF_A_MIN_WORKER_PORT="${A_MIN_WORKER_PORT}"
export PERF_A_MAX_WORKER_PORT="${A_MAX_WORKER_PORT}"
export PERF_A_COMPANY_SPU_PORT="${A_COMPANY_SPU_PORT}"
export PERF_A_COORDINATOR_SPU_PORT="${A_COORDINATOR_SPU_PORT}"
export PERF_B_IP="${B_EFFECTIVE_IP}"
export PERF_B_PARTNER_SPU_PORT="${B_PARTNER_SPU_PORT}"
export PERF_RAY_NUM_CPUS="${RAY_NUM_CPUS}"
export PERF_RAY_OBJECT_STORE_MEMORY="${RAY_OBJECT_STORE_MEMORY}"

# ------------------------------------------------------------------
# 清理 (A 本地 + B 远程)
# ------------------------------------------------------------------
cleanup() {
    echo "=== 清理 ==="
    if [ "${SINGLE_MACHINE}" = "0" ]; then
        _ssh_b "ray stop --force" 2>/dev/null || true
    fi
    conda run -n "${CONDA_ENV}" ray stop --force 2>/dev/null || true
}
trap cleanup EXIT

# ------------------------------------------------------------------
# Step 1: A 启动 Ray Head
# ------------------------------------------------------------------
echo "[1/3] A 启动 Ray Head..."

export RAY_memory_usage_threshold="${RAY_memory_usage_threshold:-0.99}"
export RAY_memory_monitor_refresh_ms="${RAY_memory_monitor_refresh_ms:-0}"

conda run -n "${CONDA_ENV}" ray stop --force 2>/dev/null || true

conda run -n "${CONDA_ENV}" ray start --head \
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
# Step 2: B 启动 Partner Worker
# ------------------------------------------------------------------
if [ "${SINGLE_MACHINE}" = "1" ]; then
    echo "[2/3] B 启动 Partner Worker (本机退化)..."
    conda run -n "${CONDA_ENV}" ray start \
        --address="${A_IP}:${RAY_PORT}" \
        --node-ip-address="${A_IP}" \
        --num-cpus="${RAY_NUM_CPUS}" \
        --object-store-memory="${RAY_OBJECT_STORE_MEMORY}" \
        --resources='{"partner": 10}' \
        --node-manager-port="${B_NODE_MANAGER_PORT}" \
        --object-manager-port="${B_OBJECT_MANAGER_PORT}" \
        --min-worker-port="${B_MIN_WORKER_PORT}" \
        --max-worker-port="${B_MAX_WORKER_PORT}"
else
    echo "[2/3] B 启动 Partner Worker (SSH ${B_SSH_USER}@${B_PUBLIC_IP})..."
    _ssh_b "ray stop --force" 2>/dev/null || true
    _ssh_b "
        export RAY_memory_usage_threshold=0.99
        export RAY_memory_monitor_refresh_ms=0
        ray start \
            --address='${A_IP}:${RAY_PORT}' \
            --node-ip-address='${B_PUBLIC_IP}' \
            --num-cpus='${RAY_NUM_CPUS}' \
            --object-store-memory='${RAY_OBJECT_STORE_MEMORY}' \
            --resources='{\"partner\": 10}' \
            --node-manager-port='${B_NODE_MANAGER_PORT}' \
            --object-manager-port='${B_OBJECT_MANAGER_PORT}' \
            --min-worker-port='${B_MIN_WORKER_PORT}' \
            --max-worker-port='${B_MAX_WORKER_PORT}'
    "
fi

# ------------------------------------------------------------------
# 等待 Partner Worker 上线
# ------------------------------------------------------------------
for i in $(seq 1 30); do
    if conda run -n "${CONDA_ENV}" ray status --address="${A_IP}:${RAY_PORT}" 2>/dev/null | grep -q partner; then
        echo "Partner Worker 在线."
        break
    fi
    sleep 2
done
if ! conda run -n "${CONDA_ENV}" ray status --address="${A_IP}:${RAY_PORT}" 2>/dev/null | grep -q partner; then
    echo "Partner Worker 未能上线, 终止."
    exit 1
fi

# ------------------------------------------------------------------
# Step 3: 运行可扩展性/推理/批量测试 (排除 WAN 和 Stress)
# ------------------------------------------------------------------
echo "[3/3] 运行可扩展性/推理/批量性能测试..."
echo "======================================"
echo " A:                  ${A_IP}"
if [ "${SINGLE_MACHINE}" = "1" ]; then
    echo " B (单机退化):       ${B_EFFECTIVE_IP}"
else
    echo " B:                  ${B_PUBLIC_IP}"
fi
echo " Conda env:          ${CONDA_ENV}"
echo "======================================"

cd "${PROJECT_ROOT}"

PYTEST_TIMEOUT_ARGS=()
if conda run -n "${CONDA_ENV}" pytest --help 2>/dev/null | grep -q -- "--timeout"; then
    PYTEST_TIMEOUT_ARGS=(--timeout=0)
fi

conda run -n "${CONDA_ENV}" pytest "${PYTEST_TARGET}" \
    -m "performance and not wan and not slow" \
    "${PYTEST_TIMEOUT_ARGS[@]}" \
    -rs "$@"

echo ""
echo "[完成] 可扩展性/推理/批量性能测试完成。"