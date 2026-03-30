#!/usr/bin/env bash
set -euo pipefail

# === 只启动 Ray head，不运行训练 ===
A_IP="192.168.1.23"
PORT_RAY="20001"

cd "$(dirname "$0")"

echo "正在启动Ray集群..."

# 启动 Ray head（只打 company/coordinator 资源）
ray start --head --node-ip-address "${A_IP}" --port "${PORT_RAY}" \
  --num-cpus 16 \
  --resources='{"company": 10, "coordinator": 10}' \
  --object-store-memory=2000000000 \
  --object-manager-port=20002 \
  --node-manager-port=20003 \
  --min-worker-port=10060 \
  --max-worker-port=10160 
echo "✅ Ray head启动成功"
echo "📍 Ray集群地址: ${A_IP}:${PORT_RAY}"

# 等待GCS服务完全启动（重要！）
echo "等待GCS服务就绪..."
sleep 3

# 验证Ray集群状态
MAX_RETRIES=10
RETRY_COUNT=0
while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if ray status >/dev/null 2>&1; then
        echo "✅ Ray集群GCS服务已就绪"
        break
    fi
    RETRY_COUNT=$((RETRY_COUNT + 1))
    echo "等待GCS服务启动... ($RETRY_COUNT/$MAX_RETRIES)"
    sleep 2
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo "❌ GCS服务启动超时"
    exit 1
fi

echo "等待Partner节点加入..."

# 保持脚本运行，等待停止信号
echo "Ray集群保持运行中，按Ctrl+C停止"
echo "提示：使用 'ray stop' 命令停止Ray集群"

# 无限等待，只有收到信号才退出
# 这样Ray会一直在后台运行
tail -f /dev/null

