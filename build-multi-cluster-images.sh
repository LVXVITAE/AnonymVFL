#!/bin/bash
# 多集群部署 - 镜像构建脚本
# 构建 Company、Partner、Coordinator 三个独立的 Docker 镜像

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 版本号
VERSION="${1:-v1.0.16}"
REGISTRY="${2:-}"  # 可选的镜像仓库地址

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}🚀 多集群部署镜像构建${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "版本: ${VERSION}"
if [ -n "$REGISTRY" ]; then
    echo "仓库: ${REGISTRY}"
fi
echo ""

# 检查 Dockerfile 是否存在
if [ ! -f "Dockerfile.company" ]; then
    echo -e "${RED}❌ 错误: Dockerfile.company 不存在${NC}"
    exit 1
fi

if [ ! -f "Dockerfile.partner" ]; then
    echo -e "${RED}❌ 错误: Dockerfile.partner 不存在${NC}"
    exit 1
fi

if [ ! -f "Dockerfile.coordinator" ]; then
    echo -e "${RED}❌ 错误: Dockerfile.coordinator 不存在${NC}"
    exit 1
fi

# 构建 Company 镜像
echo -e "${YELLOW}📦 构建 Company 镜像...${NC}"
if [ -n "$REGISTRY" ]; then
    COMPANY_IMAGE="${REGISTRY}/mobile-mpc-company:${VERSION}"
else
    COMPANY_IMAGE="mobile-mpc-company:${VERSION}"
fi

docker build -f Dockerfile.company -t ${COMPANY_IMAGE} .
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Company 镜像构建成功: ${COMPANY_IMAGE}${NC}"
else
    echo -e "${RED}❌ Company 镜像构建失败${NC}"
    exit 1
fi

# 构建 Partner 镜像
echo ""
echo -e "${YELLOW}📦 构建 Partner 镜像...${NC}"
if [ -n "$REGISTRY" ]; then
    PARTNER_IMAGE="${REGISTRY}/mobile-mpc-partner:${VERSION}"
else
    PARTNER_IMAGE="mobile-mpc-partner:${VERSION}"
fi

docker build -f Dockerfile.partner -t ${PARTNER_IMAGE} .
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Partner 镜像构建成功: ${PARTNER_IMAGE}${NC}"
else
    echo -e "${RED}❌ Partner 镜像构建失败${NC}"
    exit 1
fi

# 构建 Coordinator 镜像
echo ""
echo -e "${YELLOW}📦 构建 Coordinator 镜像...${NC}"
if [ -n "$REGISTRY" ]; then
    COORDINATOR_IMAGE="${REGISTRY}/mobile-mpc-coordinator:${VERSION}"
else
    COORDINATOR_IMAGE="mobile-mpc-coordinator:${VERSION}"
fi

docker build -f Dockerfile.coordinator -t ${COORDINATOR_IMAGE} .
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Coordinator 镜像构建成功: ${COORDINATOR_IMAGE}${NC}"
else
    echo -e "${RED}❌ Coordinator 镜像构建失败${NC}"
    exit 1
fi

# 显示镜像信息
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}✅ 镜像构建完成${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
docker images | grep "mobile-mpc-company\|mobile-mpc-partner\|mobile-mpc-coordinator" | head -3

# 保存镜像到本地文件（用于 Kubernetes 加载）
echo ""
echo -e "${YELLOW}💾 导出镜像到本地文件...${NC}"

docker save ${COMPANY_IMAGE} -o mobile-mpc-company-${VERSION}.tar
echo -e "${GREEN}✅ Company 镜像已导出: mobile-mpc-company-${VERSION}.tar${NC}"

docker save ${PARTNER_IMAGE} -o mobile-mpc-partner-${VERSION}.tar
echo -e "${GREEN}✅ Partner 镜像已导出: mobile-mpc-partner-${VERSION}.tar${NC}"

docker save ${COORDINATOR_IMAGE} -o mobile-mpc-coordinator-${VERSION}.tar
echo -e "${GREEN}✅ Coordinator 镜像已导出: mobile-mpc-coordinator-${VERSION}.tar${NC}"

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}🎉 所有操作完成！${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "镜像列表："
echo "  - ${COMPANY_IMAGE}"
echo "  - ${PARTNER_IMAGE}"
echo "  - ${COORDINATOR_IMAGE}"
echo ""
echo "导出文件："
echo "  - mobile-mpc-company-${VERSION}.tar"
echo "  - mobile-mpc-partner-${VERSION}.tar"
echo "  - mobile-mpc-coordinator-${VERSION}.tar"
echo ""
echo "后续步骤："
echo "1. 在集群 A 中加载 Company 镜像："
echo "   docker load -i mobile-mpc-company-${VERSION}.tar"
echo ""
echo "2. 在集群 B 中加载 Partner 镜像："
echo "   docker load -i mobile-mpc-partner-${VERSION}.tar"
echo ""
echo "3. 在集群 C 中加载 Coordinator 镜像："
echo "   docker load -i mobile-mpc-coordinator-${VERSION}.tar"
echo ""
echo "4. 部署到对应集群（见部署文档）"

