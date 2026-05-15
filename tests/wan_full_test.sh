#!/usr/bin/env bash
# ===== WAN 性能测试完整启动 (机器A, 有公网IP) =====
# 运行前确保机器B已执行 wan_tunnel_setup.sh 并保持运行
#
# 用法:
#   PERF_WAN_TC_REMOTE_USER=root bash tests/wan_full_test.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG="${SCRIPT_DIR}/distributed_config.yaml"

_a_ip()    { python3 -c "import yaml; c=yaml.safe_load(open('${CONFIG}')); print(c['machine_a']['ip'])"; }
_b_ip()    { python3 -c "import yaml; c=yaml.safe_load(open('${CONFIG}')); print(c['machine_b']['ip'])"; }
_a_ray()   { python3 -c "import yaml; c=yaml.safe_load(open('${CONFIG}')); print(c['machine_a']['ray_port'])"; }

A_IP=$(_a_ip)
B_IP=$(_b_ip)
RAY_PORT=$(_a_ray)

if [ "${B_IP}" != "127.0.0.1" ]; then
    echo "machine_b.ip 应为 127.0.0.1 (通过 SSH 反向隧道访问)"
    echo "当前: ${B_IP}, 请修改 ${CONFIG}"
    exit 1
fi

CONDA_ENV="${PERF_CONDA_ENV:-sf}"
TC_DEVICE="${PERF_TC_DEVICE:-eth0}"
TC_REMOTE_USER="${PERF_WAN_TC_REMOTE_USER:?请设置 PERF_WAN_TC_REMOTE_USER (B 侧 SSH 用户名)}"
TC_REMOTE_PORT="${PERF_WAN_TC_REMOTE_PORT:-2222}"
TC_REMOTE_DEVICE="${PERF_WAN_TC_REMOTE_DEVICE:-eth0}"

export PERF_RUN_WAN=1
export PERF_TC_DEVICE="${TC_DEVICE}"
export PERF_WAN_TC_REMOTE_HOST="localhost"
export PERF_WAN_TC_REMOTE_PORT="${TC_REMOTE_PORT}"
export PERF_WAN_TC_REMOTE_USER="${TC_REMOTE_USER}"
export PERF_WAN_TC_REMOTE_DEVICE="${TC_REMOTE_DEVICE}"
export PERF_WAN_CONDITIONS="${PERF_WAN_CONDITIONS:-10:20,10:50,25:20,25:50,50:20,50:50}"
export PYTHONPATH="${PROJECT_ROOT}/company:${PROJECT_ROOT}${PYTHONPATH:+:}${PYTHONPATH:-}"

cleanup() {
    echo "清理 tc (A+B)..."
    sudo tc qdisc del dev "${TC_DEVICE}" root 2>/dev/null || true
    sudo tc qdisc del dev lo root 2>/dev/null || true
    ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
        -p "${TC_REMOTE_PORT}" "${TC_REMOTE_USER}@localhost" \
        "sudo tc qdisc del dev ${TC_REMOTE_DEVICE} root" 2>/dev/null || true
}
trap cleanup EXIT

if ! ray status --address="${A_IP}:${RAY_PORT}" &>/dev/null; then
    echo "Ray Head 未运行, 请先启动: ray start --head --port=${RAY_PORT} ..."
    exit 1
fi

echo "======================================"
echo " AnonymVFL WAN 性能测试 (A+B tc)"
echo "======================================"
echo " A 网卡+tc:      ${TC_DEVICE}"
echo " B SSH:          ssh -p ${TC_REMOTE_PORT} ${TC_REMOTE_USER}@localhost"
echo " B 网卡+tc:      ${TC_REMOTE_DEVICE}"
echo " WAN:            ${PERF_WAN_CONDITIONS}"
echo " Conda env:      ${CONDA_ENV}"
echo "======================================"
echo "检查 Partner Worker..."
ray status --address="${A_IP}:${RAY_PORT}" | grep -q partner || {
    echo "Partner Worker 未检测到, 请确保机器B已执行 wan_tunnel_setup.sh"
    exit 1
}
echo "Partner Worker 在线."

cd "${PROJECT_ROOT}"

PYTEST_TIMEOUT_ARGS=()
if conda run -n "${CONDA_ENV}" pytest --help 2>/dev/null | grep -q -- "--timeout"; then
    PYTEST_TIMEOUT_ARGS=(--timeout=0)
fi

conda run -n "${CONDA_ENV}" pytest tests/test_performance.py::TestWANNetworkImpact \
    -m "performance and wan and not slow" \
    "${PYTEST_TIMEOUT_ARGS[@]}" \
    -rs "$@"

echo ""
echo "WAN 性能测试完成。结果位于 test_results/performance/"
