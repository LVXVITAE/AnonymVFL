#!/usr/bin/env bash
# 等待 WAN 测试完成后自动运行压力测试
set -euo pipefail

PROJECT_ROOT="/home/ubuntu/AnonymVFL"
LOG="/tmp/auto_stress.log"

echo "$(date '+%Y-%m-%d %H:%M:%S') 等待 WAN 测试结束..." | tee -a "$LOG"

while pgrep -f "wan_setup.sh" > /dev/null 2>&1 || pgrep -f "TestWANNetworkImpact" > /dev/null 2>&1; do
    sleep 30
done

echo "$(date '+%Y-%m-%d %H:%M:%S') WAN 测试已结束，清理环境..." | tee -a "$LOG"

conda run -n sf ray stop --force 2>/dev/null || true
sshpass -p "ubuntu" ssh -o StrictHostKeyChecking=no ubuntu@192.168.122.185 "source ~/miniconda3/etc/profile.d/conda.sh && conda activate sf && ray stop --force" 2>/dev/null || true
sleep 5

echo "$(date '+%Y-%m-%d %H:%M:%S') 启动压力测试..." | tee -a "$LOG"

cd "$PROJECT_ROOT"
bash tests/stress_setup.sh >> "$LOG" 2>&1
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') 压力测试完成！" | tee -a "$LOG"
else
    echo "$(date '+%Y-%m-%d %H:%M:%S') 压力测试失败 (exit=$EXIT_CODE)，查看日志: $LOG" | tee -a "$LOG"
fi