#!/usr/bin/env bash
set -euo pipefail

# One-click runner for WAN network impact benchmarks.
#
# Usage:
#   tests/run_wan_performance.sh
#
# Optional overrides:
#   PERF_WAN_CONDITIONS=10:20,25:50 tests/run_wan_performance.sh
#   PERF_CONDA_ENV=sf tests/run_wan_performance.sh
#   PERF_TC_DEVICE=eth0 tests/run_wan_performance.sh
#
# Uses tc/netem + tbf to simulate bandwidth and latency.
# Requires sudo for tc commands.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONDA_ENV="${PERF_CONDA_ENV:-sf}"
PYTEST_TARGET="${PERF_WAN_PYTEST_TARGET:-tests/test_performance.py::TestWANNetworkImpact}"
TC_DEVICE="${PERF_TC_DEVICE:-lo}"

export PERF_RUN_WAN=1
export PERF_WAN_CONDITIONS="${PERF_WAN_CONDITIONS:-10:20,10:50,25:20,25:50,50:20,50:50}"
export PERF_TC_DEVICE="${TC_DEVICE}"

# Verify sudo access for tc
if ! sudo -n true 2>/dev/null; then
    echo "需要 sudo 权限来配置 tc/netem。请确保当前用户有 sudo 权限。"
    exit 1
fi

cleanup() {
    sudo tc qdisc del dev "${TC_DEVICE}" root 2>/dev/null || true
}
trap cleanup EXIT

echo "======================================"
echo " AnonymVFL WAN 性能测试 (tc/netem)"
echo "======================================"
echo " Conda env:       ${CONDA_ENV}"
echo " Pytest target:   ${PYTEST_TARGET}"
echo " TC device:       ${TC_DEVICE}"
echo " WAN conditions:  ${PERF_WAN_CONDITIONS}"
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
echo "WAN 性能测试完成。结果位于 test_results/performance/:"
echo "  - psi_network_impact_wan.csv / .png"
echo "  - sslr_network_impact_wan.csv / .png"
echo "  - xgboost_network_impact_wan.csv / .png"
