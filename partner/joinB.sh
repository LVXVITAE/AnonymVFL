#!/usr/bin/env bash
set -euo pipefail

# === B = partner (worker 节点) ===
A_IP="210.28.133.104"
PORT_RAY="20001"

cd "$(dirname "$0")"

# B 加入 Ray 集群，只打 partner 资源
# 为避免与 head 端口冲突，单独指定 node-manager/object-manager/worker 端口段
# ray stop || true
ray start --address "${A_IP}:${PORT_RAY}" \
  --node-manager-port=54001 \
  --object-manager-port=54002 \
  --min-worker-port=54003 \
  --max-worker-port=54103 \
  --dashboard-port=0 \
  --num-cpus 8 \
  --resources='{"partner": 10}' \
  --object-store-memory=2000000000

# 确保必要目录存在
# mkdir -p /home/dxn/mobile_project3/partner/models/lr_partner || true
