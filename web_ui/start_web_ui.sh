#!/bin/bash
# 联邦学习Web UI启动脚本
# 支持跨机器通信的分布式联邦学习界面

set -euo pipefail

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WEB_UI_DIR="$PROJECT_ROOT/web_ui"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}    联邦学习Web UI启动脚本${NC}"
echo -e "${BLUE}    支持跨机器通信${NC}"
echo -e "${BLUE}========================================${NC}"

# 检查Python环境
echo -e "${YELLOW}检查Python环境...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}错误: 未找到python3，请先安装Python 3.7+${NC}"
    exit 1
fi

# 检查必要的Python包
echo -e "${YELLOW}检查Python依赖...${NC}"
REQUIRED_PACKAGES=("flask" "flask-socketio" "psutil" "pyyaml")
MISSING_PACKAGES=()

for package in "${REQUIRED_PACKAGES[@]}"; do
    if ! python3 -c "import $package" 2>/dev/null; then
        MISSING_PACKAGES+=("$package")
    fi
done

if [ ${#MISSING_PACKAGES[@]} -ne 0 ]; then
    echo -e "${YELLOW}安装缺失的Python包: ${MISSING_PACKAGES[*]}${NC}"
    pip3 install "${MISSING_PACKAGES[@]}"
fi

# 检查项目结构
echo -e "${YELLOW}检查项目结构...${NC}"
if [ ! -d "$PROJECT_ROOT/company" ]; then
    echo -e "${RED}错误: 未找到company目录${NC}"
    exit 1
fi

if [ ! -d "$PROJECT_ROOT/partner" ]; then
    echo -e "${RED}错误: 未找到partner目录${NC}"
    exit 1
fi

# 检查必要的脚本文件
REQUIRED_FILES=(
    "$PROJECT_ROOT/company/startA.sh"
    "$PROJECT_ROOT/company/truerun.py"
    "$PROJECT_ROOT/partner/joinB.sh"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo -e "${RED}错误: 未找到必要文件: $file${NC}"
        exit 1
    fi
done

# 设置权限
echo -e "${YELLOW}设置脚本权限...${NC}"
chmod +x "$PROJECT_ROOT/company/startA.sh"
chmod +x "$PROJECT_ROOT/company/clean.sh"
chmod +x "$PROJECT_ROOT/partner/joinB.sh"
chmod +x "$PROJECT_ROOT/partner/clean.sh"

# 创建必要的目录
echo -e "${YELLOW}创建必要目录...${NC}"
mkdir -p "$WEB_UI_DIR/templates"
mkdir -p "$WEB_UI_DIR/static"
mkdir -p "$WEB_UI_DIR/logs"

# 检查配置文件
if [ ! -f "$WEB_UI_DIR/config.yaml" ]; then
    echo -e "${YELLOW}创建默认配置文件...${NC}"
    # 配置文件已在之前创建
fi

# 显示配置信息
echo -e "${GREEN}配置信息:${NC}"
echo -e "  项目根目录: $PROJECT_ROOT"
echo -e "  Web UI目录: $WEB_UI_DIR"
echo -e "  Company路径: $PROJECT_ROOT/company"
echo -e "  Partner路径: $PROJECT_ROOT/partner"

# 显示网络配置
echo -e "${GREEN}网络配置:${NC}"
if [ -f "$WEB_UI_DIR/config.yaml" ]; then
    echo -e "  Company IP: $(grep -A1 'company:' "$WEB_UI_DIR/config.yaml" | grep 'ip:' | awk '{print $2}' | tr -d '"')"
    echo -e "  Partner IP: $(grep -A1 'partner:' "$WEB_UI_DIR/config.yaml" | grep 'ip:' | awk '{print $2}' | tr -d '"')"
    echo -e "  Ray端口: $(grep 'port_ray:' "$WEB_UI_DIR/config.yaml" | awk '{print $2}')"
fi

# 启动Web服务
echo -e "${GREEN}启动Web UI服务...${NC}"
echo -e "${BLUE}访问地址: http://localhost:5000${NC}"
echo -e "${BLUE}按 Ctrl+C 停止服务${NC}"
echo ""

cd "$WEB_UI_DIR"
python3 app.py
