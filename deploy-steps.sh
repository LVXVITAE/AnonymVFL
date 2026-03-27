#!/bin/bash
# 多集群部署脚本 - Company (minikube) + Partner (cluster-b)
# 镜像配置见 values-cluster-a.yaml / values-cluster-b.yaml

set -e

cd /home/dxn/mobile_project_final/mobile_project3_new

echo "=========================================="
echo "📊 查看 minikube 空间"
echo "=========================================="
echo "--- minikube (Company) 磁盘 ---"
sudo minikube ssh -- df -h / 2>/dev/null || true
echo ""
echo "--- minikube (Company) 镜像列表 ---"
sudo minikube image ls 2>/dev/null | grep -E "mobile-mpc|REPOSITORY" || true
echo ""
echo "--- cluster-b (Partner) 磁盘 ---"
sudo minikube ssh -p cluster-b -- df -h / 2>/dev/null || true
echo ""
echo "--- cluster-b (Partner) 镜像列表 ---"
sudo minikube image ls -p cluster-b 2>/dev/null | grep -E "mobile-mpc|REPOSITORY" || true

echo ""
echo "=========================================="
echo "📦 加载镜像到 minikube"
echo "=========================================="
echo "加载 Company 镜像到 minikube..."
sudo minikube image load mobile-mpc-company:v1.0.21
echo "加载 Partner 镜像到 cluster-b..."
sudo minikube image load mobile-mpc-partner:v1.0.20 -p cluster-b

echo ""
echo "=========================================="
echo "🗑️  卸载旧版本"
echo "=========================================="
sudo helm uninstall mobile-mpc -n mpc --kube-context minikube 2>/dev/null || true
sudo helm uninstall mobile-mpc -n mpc --kube-context cluster-b 2>/dev/null || true

echo ""
echo "=========================================="
echo "🚀 部署 Company (minikube)"
echo "=========================================="
sudo helm upgrade --install mobile-mpc ./helm-chart/mobile-mpc-project \
  -f ./helm-chart/mobile-mpc-project/values-cluster-a.yaml \
  --namespace mpc --create-namespace --kube-context minikube

echo ""
echo "=========================================="
echo "🚀 部署 Partner (cluster-b)"
echo "=========================================="
sudo helm upgrade --install mobile-mpc ./helm-chart/mobile-mpc-project \
  -f ./helm-chart/mobile-mpc-project/values-cluster-b.yaml \
  --namespace mpc --create-namespace --kube-context cluster-b

echo ""
echo "=========================================="
echo "📋 查看 Pod 状态"
echo "=========================================="
echo "--- minikube (Company) ---"
sudo kubectl --context minikube get pods -n mpc
echo ""
echo "--- cluster-b (Partner) ---"
sudo kubectl --context cluster-b get pods -n mpc

echo ""
echo "=========================================="
echo "✅ 部署完成"
echo "=========================================="
echo ""
echo "访问 WebUI:"
echo "  Company (可写): sudo kubectl --context minikube port-forward -n mpc svc/mobile-mpc-mobile-mpc-project-webui 8080:8080"
echo "  Partner (只读): sudo kubectl --context cluster-b port-forward -n mpc svc/mobile-mpc-mobile-mpc-project-webui 8081:8080"
echo ""
echo "然后访问:"
echo "  Company: http://localhost:8080"
echo "  Partner: http://localhost:8081"
