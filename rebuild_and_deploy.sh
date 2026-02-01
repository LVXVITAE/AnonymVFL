#!/bin/bash
# 重新构建和部署 MPC 项目脚本

set -e

# 配置变量（请根据实际情况修改）
IMAGE_REGISTRY="registry.example.com/mpc"
IMAGE_NAME="mobile-mpc-project"
IMAGE_TAG="v1.0.1"
NAMESPACE="mpc"
RELEASE_NAME="mpc-release"

echo "=========================================="
echo "🔨 重新构建和部署 MPC 项目"
echo "=========================================="

# 进入项目目录
cd "$(dirname "$0")"

# 1. 构建 Docker 镜像
echo ""
echo "📦 步骤 1/4: 构建 Docker 镜像..."
docker build -t ${IMAGE_REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG} .

# 2. 推送镜像到仓库
echo ""
echo "📤 步骤 2/4: 推送镜像到仓库..."
docker push ${IMAGE_REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}

# 3. 卸载旧版本
echo ""
echo "🗑️  步骤 3/4: 卸载旧版本..."
helm uninstall ${RELEASE_NAME} -n ${NAMESPACE} || true

# 等待资源清理
echo "⏳ 等待资源清理..."
sleep 5

# 4. 安装新版本
echo ""
echo "🚀 步骤 4/4: 安装新版本..."
helm install ${RELEASE_NAME} ./helm-chart/mobile-mpc-project -n ${NAMESPACE} \
  --set company.image.tag=${IMAGE_TAG} \
  --set partner.image.tag=${IMAGE_TAG}

# 5. 查看部署状态
echo ""
echo "=========================================="
echo "✅ 部署完成！查看状态："
echo "=========================================="
kubectl get pods -n ${NAMESPACE}

echo ""
echo "📝 查看 Partner 日志："
echo "kubectl logs -f -l app.kubernetes.io/component=partner -n ${NAMESPACE}"
echo ""
echo "📝 查看 Company 日志："
echo "kubectl logs -f -l app.kubernetes.io/component=company -n ${NAMESPACE}"


