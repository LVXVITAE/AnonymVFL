#!/usr/bin/env bash
# ===== WAN 隧道 + Partner Worker 启动脚本 (在机器B上运行) =====
# 机器B (内网) 通过 SSH 反向隧道暴露端口给机器A (公网IP)
#
# 用法:
#   bash tests/wan_tunnel_setup.sh <A的公网IP> <A的SSH用户>
#
# 前置: 机器A 已按 run_wan_performance.sh 启动了 Ray Head
# 前置: 机器A sshd_config 中设置 GatewayPorts yes
set -euo pipefail

A_PUBLIC_IP="${1:?请提供机器A的公网IP}"
A_SSH_USER="${2:?请提供SSH用户名}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG="${SCRIPT_DIR}/distributed_config.yaml"

# ---------- 从 config 读取端口 ----------
_a_ip()    { python3 -c "import yaml; c=yaml.safe_load(open('${CONFIG}')); print(c['machine_a']['ip'])"; }
_a_ray()   { python3 -c "import yaml; c=yaml.safe_load(open('${CONFIG}')); print(c['machine_a']['ray_port'])"; }
_a_om()    { python3 -c "import yaml; c=yaml.safe_load(open('${CONFIG}')); print(c['machine_a']['object_manager_port'])"; }
_a_nm()    { python3 -c "import yaml; c=yaml.safe_load(open('${CONFIG}')); print(c['machine_a']['node_manager_port'])"; }
_a_wmin()  { python3 -c "import yaml; c=yaml.safe_load(open('${CONFIG}')); print(c['machine_a']['min_worker_port'])"; }
_a_wmax()  { python3 -c "import yaml; c=yaml.safe_load(open('${CONFIG}')); print(c['machine_a']['max_worker_port'])"; }
_a_cspu()  { python3 -c "import yaml; c=yaml.safe_load(open('${CONFIG}')); print(c['machine_a']['company_spu_port'])"; }
_a_cospu() { python3 -c "import yaml; c=yaml.safe_load(open('${CONFIG}')); print(c['machine_a']['coordinator_spu_port'])"; }
_b_spu()   { python3 -c "import yaml; c=yaml.safe_load(open('${CONFIG}')); print(c['machine_b']['partner_spu_port'])"; }
_cpus()    { python3 -c "import yaml; c=yaml.safe_load(open('${CONFIG}')); print(c['ray']['num_cpus'])"; }
_omem()    { python3 -c "import yaml; c=yaml.safe_load(open('${CONFIG}')); print(c['ray']['object_store_memory'])"; }

A_RAY_PORT=$(_a_ray)
PARTNER_SPU_PORT=$(_b_spu)
NUM_CPUS=$(_cpus)
OBJECT_STORE_MEMORY=$(_omem)

# 映射端口: B:PORT -> A:PORT  (A需通过这些反向隧道端口访问B)
# B 端的 Ray 端口
B_NM_PORT=54001
B_OM_PORT=54002
B_WMIN_PORT=54003
B_WMAX_PORT=54103

# ---------- 生成所有 -R 参数 ----------
SSH_FORWARDS=(
    -N -T
    -o ServerAliveInterval=30
    -o ServerAliveCountMax=3
    -o ExitOnForwardFailure=yes
    -o StrictHostKeyChecking=no
    -o UserKnownHostsFile=/dev/null
    # 关键固定端口
    -R "0.0.0.0:${B_NM_PORT}:localhost:${B_NM_PORT}"
    -R "0.0.0.0:${B_OM_PORT}:localhost:${B_OM_PORT}"
    -R "0.0.0.0:${PARTNER_SPU_PORT}:localhost:${PARTNER_SPU_PORT}"
)

# 批量添加 worker 端口范围
for p in $(seq "${B_WMIN_PORT}" "${B_WMAX_PORT}"); do
    SSH_FORWARDS+=(-R "0.0.0.0:${p}:localhost:${p}")
done

echo "======================================"
echo " AnonymVFL Partner 隧道 + Worker"
echo "======================================"
echo " 机器A 公网IP:   ${A_PUBLIC_IP}"
echo " Ray Head:       ${A_PUBLIC_IP}:${A_RAY_PORT}"
echo " 转发端口:       ${B_NM_PORT}-${B_WMAX_PORT}, ${PARTNER_SPU_PORT}"
echo " CPU:            ${NUM_CPUS}"
echo "======================================"

cleanup() {
    echo "清理中..."
    ray stop --force 2>/dev/null || true
    kill %1 2>/dev/null || true
}
trap cleanup EXIT

# 1. 清理残留
ray stop --force 2>/dev/null || true

# 2. 建立 SSH 反向隧道 (通过 autossh 自动重连)
if command -v autossh &>/dev/null; then
    echo "[1/3] 建立 SSH 反向隧道 (autossh)..."
    autossh -M 0 "${SSH_FORWARDS[@]}" "${A_SSH_USER}@${A_PUBLIC_IP}" &
else
    echo "[1/3] 建立 SSH 反向隧道 (ssh, 无自动重连)..."
    ssh "${SSH_FORWARDS[@]}" "${A_SSH_USER}@${A_PUBLIC_IP}" &
fi
sleep 2

# 3. 启动 Ray Worker (以 127.0.0.1 作为 node-ip)
#    A 通过反向隧道 localhost:PORT 访问 B, 所以 B 的 node-ip 设为 127.0.0.1
echo "[2/3] 加入 Ray 集群..."
export RAY_memory_usage_threshold="${RAY_memory_usage_threshold:-0.99}"
export RAY_memory_monitor_refresh_ms="${RAY_memory_monitor_refresh_ms:-0}"

ray start \
    --address="${A_PUBLIC_IP}:${A_RAY_PORT}" \
    --node-ip-address="127.0.0.1" \
    --num-cpus="${NUM_CPUS}" \
    --object-store-memory="${OBJECT_STORE_MEMORY}" \
    --resources='{"partner": 10}' \
    --node-manager-port="${B_NM_PORT}" \
    --object-manager-port="${B_OM_PORT}" \
    --min-worker-port="${B_WMIN_PORT}" \
    --max-worker-port="${B_WMAX_PORT}"

echo "[3/3] Partner Worker 已加入集群, 保持运行..."
echo "在机器A上执行: bash tests/run_wan_performance.sh"
echo "按 Ctrl+C 退出"

# 保持运行
tail -f /dev/null
