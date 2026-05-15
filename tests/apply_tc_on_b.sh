#!/usr/bin/env bash
# ===== B 侧 tc 应用脚本 (在机器B上运行, 需要 sudo) =====
# 对 B 的物理网卡施加 tc 限制, 模拟 B→A 的 WAN 条件
#
# 用法 (由 wan_tunnel_setup.sh 自动调用, 也可单独使用):
#   sudo bash tests/apply_tc_on_b.sh <带宽MB/s> <延迟ms> <网卡>
#
# 清理:
#   sudo tc qdisc del dev <网卡> root
set -euo pipefail

BANDWIDTH="${1:?请提供带宽 MB/s}"
LATENCY="${2:?请提供延迟 ms}"
DEVICE="${3:-eth0}"

sudo tc qdisc del dev "${DEVICE}" root 2>/dev/null || true

limit=$(( LATENCY * 10 ))
[[ ${limit} -lt 100000 ]] && limit=100000

sudo tc qdisc add dev "${DEVICE}" root handle 1:0 netem delay "${LATENCY}ms" limit "${limit}"
sudo tc qdisc add dev "${DEVICE}" parent 1:0 handle 2:0 tbf rate "${BANDWIDTH}mbit" burst 32kbit latency 400ms

echo "B 侧 tc 已应用: dev=${DEVICE} bandwidth=${BANDWIDTH}Mb/s latency=${LATENCY}ms"
