#!/bin/bash
# 多集群部署脚本
# 在三个 Kubernetes 集群中分别部署 Company、Partner 和 Coordinator 节点

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 配置参数
VERSION="${VERSION:-v1.0.16}"
NAMESPACE="${NAMESPACE:-mpc-test}"
CHART_PATH="./helm-chart/mobile-mpc-project"

# 集群上下文（需要根据实际情况修改）
CLUSTER_A_CONTEXT="${CLUSTER_A_CONTEXT:-cluster-a}"
CLUSTER_B_CONTEXT="${CLUSTER_B_CONTEXT:-cluster-b}"
CLUSTER_C_CONTEXT="${CLUSTER_C_CONTEXT:-cluster-c}"

# 节点 IP（需要根据实际情况修改）
CLUSTER_A_NODE_IP="${CLUSTER_A_NODE_IP:-}"
CLUSTER_B_NODE_IP="${CLUSTER_B_NODE_IP:-}"
CLUSTER_C_NODE_IP="${CLUSTER_C_NODE_IP:-}"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}🚀 多集群联邦学习部署${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# 检查必要参数
if [ -z "$CLUSTER_A_NODE_IP" ] || [ -z "$CLUSTER_B_NODE_IP" ] || [ -z "$CLUSTER_C_NODE_IP" ]; then
    echo -e "${RED}❌ 错误: 需要设置集群节点 IP${NC}"
    echo ""
    echo "使用方法:"
    echo "  export CLUSTER_A_NODE_IP=192.168.1.100"
    echo "  export CLUSTER_B_NODE_IP=192.168.1.101"
    echo "  export CLUSTER_C_NODE_IP=192.168.1.102"
    echo "  $0"
    echo ""
    echo "或者:"
    echo "  CLUSTER_A_NODE_IP=192.168.1.100 CLUSTER_B_NODE_IP=192.168.1.101 CLUSTER_C_NODE_IP=192.168.1.102 $0"
    exit 1
fi

echo "配置信息:"
echo "  版本: ${VERSION}"
echo "  命名空间: ${NAMESPACE}"
echo "  集群 A 上下文: ${CLUSTER_A_CONTEXT}"
echo "  集群 A 节点 IP: ${CLUSTER_A_NODE_IP}"
echo "  集群 B 上下文: ${CLUSTER_B_CONTEXT}"
echo "  集群 B 节点 IP: ${CLUSTER_B_NODE_IP}"
echo "  集群 C 上下文: ${CLUSTER_C_CONTEXT}"
echo "  集群 C 节点 IP: ${CLUSTER_C_NODE_IP}"
echo ""

# 函数: 检查集群连接
check_cluster() {
    local context=$1
    echo -e "${YELLOW}🔍 检查集群连接: ${context}${NC}"
    if kubectl --context=${context} cluster-info &>/dev/null; then
        echo -e "${GREEN}✅ 集群 ${context} 连接正常${NC}"
        return 0
    else
        echo -e "${RED}❌ 无法连接到集群 ${context}${NC}"
        return 1
    fi
}

# 函数: 加载镜像到集群
load_image_to_cluster() {
    local context=$1
    local image_file=$2
    local image_name=$3
    
    echo -e "${YELLOW}📦 加载镜像到集群 ${context}: ${image_name}${NC}"
    
    # 获取集群节点
    local nodes=$(kubectl --context=${context} get nodes -o jsonpath='{.items[*].metadata.name}')
    
    for node in $nodes; do
        echo "  加载到节点: ${node}"
        # 这里需要根据实际情况调整，可能需要 ssh 或 docker save/load
        # 示例: 使用 minikube
        if command -v minikube &> /dev/null; then
            minikube --profile=${context} image load ${image_file}
        else
            # 普通 Kubernetes 集群，需要在每个节点上执行 docker load
            echo -e "${YELLOW}  请手动在节点 ${node} 上执行: docker load -i ${image_file}${NC}"
        fi
    done
}

# 函数: 部署到集群
deploy_to_cluster() {
    local context=$1
    local values_file=$2
    local release_name=$3
    
    echo -e "${YELLOW}🚀 部署到集群 ${context}${NC}"
    
    # 创建命名空间
    kubectl --context=${context} create namespace ${NAMESPACE} --dry-run=client -o yaml | kubectl --context=${context} apply -f -
    
    # 使用 Helm 部署
    helm --kube-context=${context} upgrade --install ${release_name} ${CHART_PATH} \
        --namespace ${NAMESPACE} \
        --values ${values_file} \
        --set company.image.tag=${VERSION} \
        --set partner.image.tag=${VERSION} \
        --set coordinator.image.tag=${VERSION} \
        --set webui.image.tag=${VERSION}
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ 部署成功${NC}"
    else
        echo -e "${RED}❌ 部署失败${NC}"
        return 1
    fi
}

# 主流程
echo -e "${BLUE}========== 第 1 步: 检查集群连接 ==========${NC}"
check_cluster ${CLUSTER_A_CONTEXT} || exit 1
check_cluster ${CLUSTER_B_CONTEXT} || exit 1
check_cluster ${CLUSTER_C_CONTEXT} || exit 1

echo ""
echo -e "${BLUE}========== 第 2 步: 加载镜像 ==========${NC}"
if [ -f "mobile-mpc-company-${VERSION}.tar" ]; then
    load_image_to_cluster ${CLUSTER_A_CONTEXT} "mobile-mpc-company-${VERSION}.tar" "mobile-mpc-company:${VERSION}"
else
    echo -e "${YELLOW}⚠️  镜像文件不存在，跳过加载（假设镜像已存在）${NC}"
fi

if [ -f "mobile-mpc-partner-${VERSION}.tar" ]; then
    load_image_to_cluster ${CLUSTER_B_CONTEXT} "mobile-mpc-partner-${VERSION}.tar" "mobile-mpc-partner:${VERSION}"
else
    echo -e "${YELLOW}⚠️  镜像文件不存在，跳过加载（假设镜像已存在）${NC}"
fi

if [ -f "mobile-mpc-coordinator-${VERSION}.tar" ]; then
    load_image_to_cluster ${CLUSTER_C_CONTEXT} "mobile-mpc-coordinator-${VERSION}.tar" "mobile-mpc-coordinator:${VERSION}"
else
    echo -e "${YELLOW}⚠️  Coordinator 镜像文件不存在，跳过加载（假设镜像已存在）${NC}"
fi

echo ""
echo -e "${BLUE}========== 第 3 步: 更新配置文件中的 IP 地址 ==========${NC}"

# 创建临时配置文件
cp helm-chart/mobile-mpc-project/values-cluster-a.yaml /tmp/values-cluster-a-${VERSION}.yaml
cp helm-chart/mobile-mpc-project/values-cluster-b.yaml /tmp/values-cluster-b-${VERSION}.yaml
cp helm-chart/mobile-mpc-project/values-cluster-c.yaml /tmp/values-cluster-c-${VERSION}.yaml

# 替换 IP 地址
sed -i "s/CLUSTER_B_NODE_IP/${CLUSTER_B_NODE_IP}/g" /tmp/values-cluster-a-${VERSION}.yaml
sed -i "s/CLUSTER_C_NODE_IP/${CLUSTER_C_NODE_IP}/g" /tmp/values-cluster-a-${VERSION}.yaml
sed -i "s/CLUSTER_A_NODE_IP/${CLUSTER_A_NODE_IP}/g" /tmp/values-cluster-b-${VERSION}.yaml
sed -i "s/CLUSTER_B_NODE_IP/${CLUSTER_B_NODE_IP}/g" /tmp/values-cluster-b-${VERSION}.yaml
sed -i "s/CLUSTER_C_NODE_IP/${CLUSTER_C_NODE_IP}/g" /tmp/values-cluster-b-${VERSION}.yaml
sed -i "s/CLUSTER_A_NODE_IP/${CLUSTER_A_NODE_IP}/g" /tmp/values-cluster-c-${VERSION}.yaml

echo -e "${GREEN}✅ 配置文件已更新${NC}"

echo ""
echo -e "${BLUE}========== 第 4 步: 部署到集群 A (Company) ==========${NC}"
deploy_to_cluster ${CLUSTER_A_CONTEXT} "/tmp/values-cluster-a-${VERSION}.yaml" "mpc-company" || exit 1

echo ""
echo -e "${BLUE}========== 第 5 步: 部署到集群 B (Partner) ==========${NC}"
deploy_to_cluster ${CLUSTER_B_CONTEXT} "/tmp/values-cluster-b-${VERSION}.yaml" "mpc-partner" || exit 1

echo ""
echo -e "${BLUE}========== 第 6 步: 部署到集群 C (Coordinator) ==========${NC}"
deploy_to_cluster ${CLUSTER_C_CONTEXT} "/tmp/values-cluster-c-${VERSION}.yaml" "mpc-coordinator" || exit 1

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}🎉 多集群部署完成！${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "访问信息:"
echo ""
echo "集群 A (Company):"
echo "  Web UI: http://${CLUSTER_A_NODE_IP}:30080"
echo "  查看状态: kubectl --context=${CLUSTER_A_CONTEXT} -n ${NAMESPACE} get pods"
echo ""
echo "集群 B (Partner):"
echo "  Web UI: http://${CLUSTER_B_NODE_IP}:30081 (只读)"
echo "  查看状态: kubectl --context=${CLUSTER_B_CONTEXT} -n ${NAMESPACE} get pods"
echo ""
echo "集群 C (Coordinator):"
echo "  查看状态: kubectl --context=${CLUSTER_C_CONTEXT} -n ${NAMESPACE} get pods"
echo ""
echo "查看日志:"
echo "  Company: kubectl --context=${CLUSTER_A_CONTEXT} -n ${NAMESPACE} logs -l app.kubernetes.io/component=company -f"
echo "  Partner: kubectl --context=${CLUSTER_B_CONTEXT} -n ${NAMESPACE} logs -l app.kubernetes.io/component=partner -f"
echo "  Coordinator: kubectl --context=${CLUSTER_C_CONTEXT} -n ${NAMESPACE} logs -l app.kubernetes.io/component=coordinator -f"

