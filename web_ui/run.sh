#!/bin/bash

# 完整的从头开始流程

echo "🧹 步骤1：清理环境..."
ray stop 2>/dev/null
pkill -f "start_ray_with_log" 2>/dev/null
pkill -f "join_with_log" 2>/dev/null
pkill -f "start_training_with_log" 2>/dev/null
pkill -f "app.py" 2>/dev/null

echo "🗑️  步骤2：删除旧日志..."
# 使用相对路径（基于项目根目录）
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
rm -f "${SCRIPT_DIR}/company/company_run.log"
rm -f "${SCRIPT_DIR}/partner/partner_run.log"
rm -f "${SCRIPT_DIR}/company/spu.log"
rm -f "${SCRIPT_DIR}/partner/spu.log"
rm -f "${SCRIPT_DIR}/company/partner_share.csv"

echo ""
echo "✅ 环境已清理完毕！"
echo ""
echo "🚀 步骤3：启动Web UI..."
echo "   请在新终端执行："
echo ""
echo "   cd ${SCRIPT_DIR}/web_ui"
echo "   python3 app.py"
echo ""
echo "🌐 步骤4：打开浏览器"
echo "   访问：http://127.0.0.1:5000"
echo "   按F12打开Console"
echo "   按Ctrl+Shift+R强制刷新"
echo ""
echo "📋 步骤5：按顺序点击"
echo "   1. 启动Company（等3-5秒）"
echo "   2. 启动Partner（等3-5秒）"
echo "   3. 开始训练"
echo ""
echo "🎊 祝测试成功！"


