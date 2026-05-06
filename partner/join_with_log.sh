#!/usr/bin/env bash
set -euo pipefail

# === Partner加入并输出日志 ===
RAY_HEAD_ADDR="${RAY_HEAD_ADDR:-210.28.133.104:20001}"

cd "$(dirname "$0")"
LOG_FILE="./partner_run.log"

# 分隔符（不清空，方便查看历史）
echo "" >> "$LOG_FILE"
echo "==================== 新的启动 ====================" >> "$LOG_FILE"
echo "$(date '+%Y-%m-%d %H:%M:%S') [INFO] Partner开始加入Ray集群" | tee -a "$LOG_FILE"

# 加入Ray集群
ray start --address "${RAY_HEAD_ADDR}" \
  --node-manager-port=54001 \
  --object-manager-port=54002 \
  --min-worker-port=54003 \
  --max-worker-port=54103 \
  --dashboard-port=0 \
  --num-cpus 8 \
  --resources='{"partner": 10}' \
  --object-store-memory=2000000000 2>&1 | while IFS= read -r line; do
    echo "$(date '+%Y-%m-%d %H:%M:%S') [Partner] $line" | tee -a "$LOG_FILE"
done

echo "$(date '+%Y-%m-%d %H:%M:%S') [INFO] Partner已加入Ray集群: ${RAY_HEAD_ADDR}" | tee -a "$LOG_FILE"

# 保持脚本运行（否则Partner进程会退出，导致Ray节点离开集群）
echo "$(date '+%Y-%m-%d %H:%M:%S') [INFO] Partner节点保持运行中..." | tee -a "$LOG_FILE"
tail -f /dev/null
