#!/bin/bash
# 模型参数展示功能测试脚本

echo "========================================="
echo "🧪 模型参数展示功能测试"
echo "========================================="
echo ""

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# 检查模型目录
echo "1️⃣  检查模型目录..."
if [ -d "company/models" ] && [ -d "partner/models" ]; then
    echo -e "${GREEN}✅ 模型目录存在${NC}"
    echo ""
    echo "Company 模型:"
    ls -lh company/models/*/
    echo ""
    echo "Partner 模型:"
    ls -lh partner/models/*/
else
    echo -e "${RED}❌ 模型目录不存在${NC}"
    echo "请先运行训练生成模型文件"
    exit 1
fi

echo ""
echo "2️⃣  检查 Web UI 修改..."

# 检查 app.py API
if grep -q "/api/models/list" web_ui/app.py; then
    echo -e "${GREEN}✅ API 路由已添加${NC}"
else
    echo -e "${RED}❌ API 路由未找到${NC}"
    exit 1
fi

# 检查 HTML 修改
if grep -q "models-container" web_ui/templates/index.html; then
    echo -e "${GREEN}✅ HTML 模型区域已添加${NC}"
else
    echo -e "${RED}❌ HTML 修改未找到${NC}"
    exit 1
fi

# 检查 JavaScript 函数
if grep -q "function loadModels" web_ui/templates/index.html; then
    echo -e "${GREEN}✅ JavaScript 功能已添加${NC}"
else
    echo -e "${RED}❌ JavaScript 未找到${NC}"
    exit 1
fi

echo ""
echo "3️⃣  模拟 API 调用测试..."

# 如果 Web UI 正在运行，测试 API
if curl -s http://localhost:8080/api/status > /dev/null 2>&1; then
    echo -e "${YELLOW}Web UI 正在运行，测试 API...${NC}"
    
    # 测试获取模型列表
    echo ""
    echo "测试: GET /api/models/list"
    curl -s http://localhost:8080/api/models/list | python3 -m json.tool || echo "API 调用失败"
    
else
    echo -e "${YELLOW}⚠️  Web UI 未运行，跳过 API 测试${NC}"
    echo "请启动 Web UI 后测试:"
    echo "  cd web_ui && python3 app.py"
fi

echo ""
echo "========================================="
echo "✅ 测试完成！"
echo "========================================="
echo ""
echo "📋 下一步操作："
echo "1. 启动 Web UI:"
echo "   cd web_ui && python3 app.py"
echo ""
echo "2. 打开浏览器访问:"
echo "   http://localhost:8080"
echo ""
echo "3. 滚动到页面底部查看 '训练模型参数' 区域"
echo ""
echo "4. 测试以下功能:"
echo "   - 查看模型列表"
echo "   - 点击 '查看' 按钮查看文件内容"
echo "   - 点击 '下载' 按钮下载文件"
echo "   - 点击 '刷新' 按钮刷新列表"
echo ""
