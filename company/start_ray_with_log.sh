#!/usr/bin/env bash
set -euo pipefail

# === 只启动 Company 所在机器的 Ray head，输出到日志 ===
RAY_HEAD_HOST="${RAY_HEAD_HOST:-210.28.133.104}"
RAY_PORT="${RAY_PORT:-20001}"

cd "$(dirname "$0")"
LOG_FILE="./company_run.log"

# 分隔符（不清空，方便查看历史）
echo "" >> "$LOG_FILE"
echo "==================== 新的启动 ====================" >> "$LOG_FILE"
echo "$(date '+%Y-%m-%d %H:%M:%S') [INFO] 开始启动Ray集群" | tee -a "$LOG_FILE"

# 启动 Ray head
ray start --head --node-ip-address "${RAY_HEAD_HOST}" --port "${RAY_PORT}" \
  --num-cpus 8 \
  --resources='{"company": 10}' \
  --object-store-memory=2000000000 2>&1 | while IFS= read -r line; do
    echo "$(date '+%Y-%m-%d %H:%M:%S') [Company] $line" | tee -a "$LOG_FILE"
done

echo "$(date '+%Y-%m-%d %H:%M:%S') [INFO] Ray集群已启动，地址: ${RAY_HEAD_HOST}:${RAY_PORT}" | tee -a "$LOG_FILE"
echo "$(date '+%Y-%m-%d %H:%M:%S') [INFO] 等待 Partner 与 Coordinator 机器加入 Ray 集群" | tee -a "$LOG_FILE"

# 保持运行
tail -f /dev/null

