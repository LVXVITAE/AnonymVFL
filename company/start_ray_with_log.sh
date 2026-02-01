#!/usr/bin/env bash
set -euo pipefail

# === 只启动Ray，输出到日志 ===
A_IP="210.28.133.104"
PORT_RAY="20001"

cd "$(dirname "$0")"
LOG_FILE="./company_run.log"

# 分隔符（不清空，方便查看历史）
echo "" >> "$LOG_FILE"
echo "==================== 新的启动 ====================" >> "$LOG_FILE"
echo "$(date '+%Y-%m-%d %H:%M:%S') [INFO] 开始启动Ray集群" | tee -a "$LOG_FILE"

# 启动 Ray head
ray start --head --node-ip-address "${A_IP}" --port "${PORT_RAY}" \
  --num-cpus 8 \
  --resources='{"company": 10, "coordinator": 10}' \
  --object-store-memory=2000000000 2>&1 | while IFS= read -r line; do
    echo "$(date '+%Y-%m-%d %H:%M:%S') [Company] $line" | tee -a "$LOG_FILE"
done

echo "$(date '+%Y-%m-%d %H:%M:%S') [INFO] Ray集群已启动，等待Partner加入" | tee -a "$LOG_FILE"

# 保持运行
tail -f /dev/null

