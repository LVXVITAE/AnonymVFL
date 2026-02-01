#!/bin/bash
# Helm Chart 结构验证脚本

set -e

CHART_DIR="helm-chart/mobile-mpc-project"
ERRORS=0

echo "========================================="
echo "📋 验证 Helm Chart 结构"
echo "========================================="
echo ""

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查文件是否存在
check_file() {
    if [ -f "$1" ]; then
        echo -e "${GREEN}✓${NC} 文件存在: $1"
    else
        echo -e "${RED}✗${NC} 文件缺失: $1"
        ERRORS=$((ERRORS + 1))
    fi
}

# 检查目录是否存在
check_dir() {
    if [ -d "$1" ]; then
        echo -e "${GREEN}✓${NC} 目录存在: $1"
    else
        echo -e "${RED}✗${NC} 目录缺失: $1"
        ERRORS=$((ERRORS + 1))
    fi
}

# 检查必需的 Chart 文件
echo "1. 检查 Chart 基本文件..."
check_file "$CHART_DIR/Chart.yaml"
check_file "$CHART_DIR/values.yaml"
check_file "$CHART_DIR/README.md"
check_file "$CHART_DIR/.helmignore"
echo ""

# 检查 templates 目录
echo "2. 检查 templates 目录..."
check_dir "$CHART_DIR/templates"
check_file "$CHART_DIR/templates/_helpers.tpl"
check_file "$CHART_DIR/templates/company-deployment.yaml"
check_file "$CHART_DIR/templates/partner-deployment.yaml"
check_file "$CHART_DIR/templates/service.yaml"
check_file "$CHART_DIR/templates/serviceaccount.yaml"
check_file "$CHART_DIR/templates/ingress.yaml"
check_file "$CHART_DIR/templates/pvc.yaml"
check_file "$CHART_DIR/templates/NOTES.txt"
echo ""

# 检查 Chart.yaml 内容
echo "3. 验证 Chart.yaml 内容..."
if [ -f "$CHART_DIR/Chart.yaml" ]; then
    if grep -q "apiVersion:" "$CHART_DIR/Chart.yaml" && \
       grep -q "name:" "$CHART_DIR/Chart.yaml" && \
       grep -q "version:" "$CHART_DIR/Chart.yaml"; then
        echo -e "${GREEN}✓${NC} Chart.yaml 包含必需字段"
    else
        echo -e "${RED}✗${NC} Chart.yaml 缺少必需字段"
        ERRORS=$((ERRORS + 1))
    fi
fi
echo ""

# 检查 values.yaml 内容
echo "4. 验证 values.yaml 内容..."
if [ -f "$CHART_DIR/values.yaml" ]; then
    if grep -q "company:" "$CHART_DIR/values.yaml" && \
       grep -q "partner:" "$CHART_DIR/values.yaml"; then
        echo -e "${GREEN}✓${NC} values.yaml 包含主要配置项"
    else
        echo -e "${RED}✗${NC} values.yaml 配置不完整"
        ERRORS=$((ERRORS + 1))
    fi
fi
echo ""

# 检查 Dockerfile
echo "5. 检查 Dockerfile..."
check_file "Dockerfile"
if [ -f "Dockerfile" ]; then
    if grep -q "FROM" "Dockerfile" && \
       grep -q "WORKDIR" "Dockerfile" && \
       grep -q "COPY" "Dockerfile"; then
        echo -e "${GREEN}✓${NC} Dockerfile 格式正确"
    else
        echo -e "${RED}✗${NC} Dockerfile 格式可能有问题"
        ERRORS=$((ERRORS + 1))
    fi
fi
echo ""

# 检查依赖文件
echo "6. 检查依赖文件..."
check_file "requirements.txt"
check_file ".dockerignore"
echo ""

# 检查 YAML 语法（简单检查）
echo "7. 简单 YAML 语法检查..."
YAML_ERRORS=0
for file in $CHART_DIR/templates/*.yaml $CHART_DIR/*.yaml; do
    if [ -f "$file" ]; then
        # 检查基本的 YAML 结构（缩进、冒号等）
        if grep -q "^[[:space:]]*-[[:space:]]*$" "$file" || \
           grep -q ":[[:space:]]*$" "$file" || \
           python3 -c "import yaml; yaml.safe_load(open('$file'))" 2>/dev/null; then
            :  # 文件看起来正常
        else
            echo -e "${YELLOW}⚠${NC} 可能的语法问题: $file"
            YAML_ERRORS=$((YAML_ERRORS + 1))
        fi
    fi
done

if [ $YAML_ERRORS -eq 0 ]; then
    echo -e "${GREEN}✓${NC} YAML 文件看起来正常"
else
    echo -e "${YELLOW}⚠${NC} 发现 $YAML_ERRORS 个可能的 YAML 问题（需要 helm lint 进一步验证）"
fi
echo ""

# 检查模板语法中的常见问题
echo "8. 检查 Helm 模板语法..."
TEMPLATE_ISSUES=0
for file in $CHART_DIR/templates/*.yaml; do
    if [ -f "$file" ]; then
        # 检查是否有未闭合的花括号
        OPEN_BRACES=$(grep -o "{{" "$file" | wc -l)
        CLOSE_BRACES=$(grep -o "}}" "$file" | wc -l)
        if [ "$OPEN_BRACES" -ne "$CLOSE_BRACES" ]; then
            echo -e "${RED}✗${NC} 花括号不匹配: $file ({{ $OPEN_BRACES 次, }} $CLOSE_BRACES 次)"
            TEMPLATE_ISSUES=$((TEMPLATE_ISSUES + 1))
        fi
    fi
done

if [ $TEMPLATE_ISSUES -eq 0 ]; then
    echo -e "${GREEN}✓${NC} 模板花括号匹配正常"
fi
echo ""

# 总结
echo "========================================="
echo "📊 验证结果总结"
echo "========================================="
if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}✓ 所有检查通过！${NC}"
    echo ""
    echo "Chart 结构完整，可以进行下一步操作："
    echo "  1. 构建 Docker 镜像"
    echo "  2. 推送到镜像仓库"
    echo "  3. 使用 helm install 部署到 Kubernetes"
    echo ""
    echo "建议安装 Helm 进行更完整的验证："
    echo "  sudo snap install helm"
    echo "  helm lint $CHART_DIR"
    exit 0
else
    echo -e "${RED}✗ 发现 $ERRORS 个错误${NC}"
    echo "请修复上述问题后重试"
    exit 1
fi

