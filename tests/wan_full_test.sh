#!/usr/bin/env bash
# ===== WAN 性能测试完整启动 (机器A, 有公网IP) =====
# 运行前确保机器B已执行 wan_tunnel_setup.sh 并保持运行
#
# 用法:
#   PERF_TC_DEVICE=eth0 bash tests/wan_full_test.sh
#
# tc 打在物理网卡上模拟 B→A 的 WAN 条件,
# A→B 流量走 SSH 反向隧道 (通过 lo), 不受 tc 影响.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG="${SCRIPT_DIR}/distributed_config.yaml"

# 从 config 读 machine_a.ip, 确认不是占位符
A_IP=$(python3 -c "import yaml; c=yaml.safe_load(open('${CONFIG}')); print(c['machine_a']['ip'])")
B_IP=$(python3 -c "import yaml; c=yaml.safe_load(open('${CONFIG}')); print(c['machine_b']['ip'])")
RAY_PORT=$(python3 -c "import yaml; c=yaml.safe_load(open('${CONFIG}')); print(c['machine_a']['ray_port'])")

if [ "${B_IP}" != "127.0.0.1" ]; then
    echo "WAN 隧道模式下 machine_b.ip 应设为 127.0.0.1 (通过 SSH 反向隧道访问)"
    echo "当前 machine_b.ip = ${B_IP}, 请修改 ${CONFIG}"
    exit 1
fi

CONDA_ENV="${PERF_CONDA_ENV:-sf}"
TC_DEVICE="${PERF_TC_DEVICE:-eth0}"
export PERF_RUN_WAN=1
export PERF_TC_DEVICE="${TC_DEVICE}"
export PERF_WAN_CONDITIONS="${PERF_WAN_CONDITIONS:-10:20,10:50,25:20,25:50,50:20,50:50}"

cleanup() {
    echo "清理 tc..."
    sudo tc qdisc del dev "${TC_DEVICE}" root 2>/dev/null || true
    sudo tc qdisc del dev lo root 2>/dev/null || true
}
trap cleanup EXIT

if ! ray status --address="${A_IP}:${RAY_PORT}" &>/dev/null; then
    echo "Ray Head 未运行, 请在机器A上先启动: ray start --head --port=${RAY_PORT} ..."
    exit 1
fi

echo "======================================"
echo " AnonymVFL WAN 性能测试 (跨机 + tc)"
echo "======================================"
echo " 机器A IP:       ${A_IP}"
echo " 机器B 隧道:     localhost (反向隧道)"
echo " TC device:      ${TC_DEVICE}"
echo " WAN conditions: ${PERF_WAN_CONDITIONS}"
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
