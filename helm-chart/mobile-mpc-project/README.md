# Mobile MPC Project Helm Chart

这是一个基于 SecretFlow 的移动多方安全计算（MPC）项目的 Kubernetes Helm Chart，支持**单集群**和**多集群**两种部署模式。

## 简介

此 Chart 用于在 Kubernetes 集群中部署多方安全计算应用，支持：

- **Company 节点**（甲方/服务器）：运行 Ray Head 和 MPC 训练任务
- **Partner 节点**（乙方/客户端）：运行 Ray Worker 参与联合训练
- **Coordinator 节点**（可选协调者）：协调多方计算流程
- **Web UI 管理界面**：提供训练任务管理、模型查看、日志监控等功能

### 部署模式

#### 单集群部署
- 所有组件（Company、Partner、WebUI）部署在同一 Kubernetes 集群
- 使用 Kubernetes Service DNS 进行服务发现
- 适用于开发和测试环境

#### 多集群部署（跨集群）
- Company 和 Partner 分别部署在不同的 Kubernetes 集群
- 使用 `hostNetwork` 模式实现跨集群直接通信
- 适用于生产环境，满足数据隔离和合规要求

## 前置要求

### 基础要求
- Kubernetes 1.19+
- Helm 3.2.0+
- kubectl 已配置并可以访问目标集群

### 镜像要求
- 已构建的 Docker 镜像：
  - `mobile-mpc-company:v1.0.16`（Company 节点镜像，包含 WebUI）
  - `mobile-mpc-partner:v1.0.16`（Partner 节点镜像，包含只读 WebUI）

### 多集群部署额外要求
- 两个独立的 Kubernetes 集群
- 集群间网络互通（节点 IP 可访问）
- 了解各集群的节点 IP 地址

## 快速开始

### 1. 构建 Docker 镜像

cd /path/to/mobile_project3_new

# 构建 Company 镜像
docker build -t mobile-mpc-company:v1.0.16 -f Dockerfile.company .

# 构建 Partner 镜像
docker build -t mobile-mpc-partner:v1.0.16 -f Dockerfile.partner .### 2. 加载镜像到 Minikube（如果使用 Minikube）

# 加载到 minikube 集群
docker save mobile-mpc-company:v1.0.16 | minikube image load -
docker save mobile-mpc-partner:v1.0.16 | minikube image load -

# 加载到 cluster-b（多集群场景）
docker save mobile-mpc-partner:v1.0.16 | minikube image load - -p cluster-b## 部署方式

### 方式一：单集群部署

适用于所有组件在同一集群的场景。

#### 1. 创建命名空间

kubectl create namespace mpc-test#### 2. 安装 Chart

helm install mpc-test ./helm-chart/mobile-mpc-project \
  -n mpc-test \
  -f values.yaml#### 3. 验证部署

# 查看 Pod 状态
kubectl get pods -n mpc-test

# 查看服务
kubectl get svc -n mpc-test

# 访问 WebUI
kubectl port-forward -n mpc-test svc/mpc-test-mobile-mpc-project-webui 8080:8080然后访问 `http://localhost:8080`

---

### 方式二：多集群部署（跨集群）

适用于 Company 和 Partner 部署在不同集群的场景。

#### 前置准备

1. **获取节点 IP 地址**
h
# 获取集群 A (Company) 节点 IP
kubectl config use-context minikube
kubectl get nodes -o wide

# 获取集群 B (Partner) 节点 IP
kubectl config use-context cluster-b
kubectl get nodes -o wide2. **更新配置文件**

编辑 `values-cluster-a.yaml` 和 `values-cluster-b.yaml`，替换节点 IP：

# values-cluster-a.yaml
env:
  - name: PARTNER_SPU_ADDR
    value: "192.168.58.2:9395"  # 替换为集群 B 节点 IP

# values-cluster-b.yaml
env:
  - name: COMPANY_SPU_ADDR
    value: "192.168.49.2:9394"  # 替换为集群 A 节点 IP
  - name: COORDINATOR_SPU_ADDR
    value: "192.168.49.2:9396"
  - name: RAY_HEAD_ADDR
    value: "192.168.49.2:6379"#### 部署步骤

**1. 部署 Company 集群（集群 A）**

kubectl config use-context minikube
kubectl create namespace mpc-test

helm install mpc-company ./helm-chart/mobile-mpc-project \
  -n mpc-test \
  -f values-cluster-a.yaml**2. 部署 Partner 集群（集群 B）**

kubectl config use-context cluster-b
kubectl create namespace mpc-test

helm install mpc-partner ./helm-chart/mobile-mpc-project \
  -n mpc-test \
  -f values-cluster-b.yaml**3. 验证部署**

# 检查 Company 集群
kubectl config use-context minikube
kubectl get pods -n mpc-test
kubectl logs -n mpc-test -l app.kubernetes.io/component=company

# 检查 Partner 集群
kubectl config use-context cluster-b
kubectl get pods -n mpc-test
kubectl logs -n mpc-test -l app.kubernetes.io/component=partner**4. 访问 WebUI**

# Company WebUI（可写）
kubectl config use-context minikube
kubectl port-forward -n mpc-test svc/mpc-company-mobile-mpc-project-webui 8080:8080

# Partner WebUI（只读）
kubectl config use-context cluster-b
kubectl port-forward -n mpc-test svc/mpc-partner-mobile-mpc-project-webui 8081:8080
## 配置说明

### 主要配置项

| 参数 | 描述 | 默认值 | 单集群 | 多集群 |
|------|------|--------|--------|--------|
| `company.enabled` | 是否启用 Company 节点 | `true` | ✅ | ✅ (仅集群 A) |
| `partner.enabled` | 是否启用 Partner 节点 | `true` | ✅ | ✅ (仅集群 B) |
| `webui.enabled` | 是否启用 Web UI | `true` | ✅ | ✅ |
| `company.service.type` | Company 服务类型 | `ClusterIP` | `ClusterIP` | `NodePort` |
| `partner.service.type` | Partner 服务类型 | `ClusterIP` | `ClusterIP` | `NodePort` |
| `partner.crossCluster.enabled` | 启用跨集群通信 | `false` | ❌ | ✅ |
| `persistence.enabled` | 启用持久化存储 | `false` | 可选 | 推荐 |

### 网络配置

#### 单集群模式

- **Service 类型**：`ClusterIP`
- **通信方式**：Kubernetes Service DNS
- **地址解析**：自动通过 Service 名称解析 Pod IP

#### 多集群模式

- **Service 类型**：`NodePort`（配置保留，实际使用 hostNetwork）
- **通信方式**：节点 IP 直接通信
- **hostNetwork**：`true`（Pod 使用节点网络）
- **地址配置**：通过环境变量指定跨集群节点 IP

**关键配置示例：**
aml
# values-cluster-a.yaml
company:
  service:
    type: NodePort
  env:
    - name: PARTNER_SPU_ADDR
      value: "192.168.58.2:9395"  # 集群 B 节点 IP

# values-cluster-b.yaml
partner:
  crossCluster:
    enabled: true
  service:
    type: NodePort
  env:
    - name: COMPANY_SPU_ADDR
      value: "192.168.49.2:9394"  # 集群 A 节点 IP
    - name: RAY_HEAD_ADDR
      value: "192.168.49.2:6379"### 资源限制

company:
  resources:
    requests:
      cpu: "8"
      memory: "4Gi"
    limits:
      cpu: "12"
      memory: "8Gi"

partner:
  resources:
    requests:
      cpu: "8"
      memory: "4Gi"
    limits:
      cpu: "12"
      memory: "8Gi"### 持久化存储

persistence:
  enabled: true
  path: "/data/mobile-mpc"  # 宿主机路径（hostPath）
  storageClass: ""  # 空字符串表示使用 hostPath## 常用命令

### 查看状态

# 查看 Helm 发布状态
helm status mpc-test -n mpc-test

# 查看所有资源
kubectl get all -n mpc-test

# 查看 Pod 状态
kubectl get pods -n mpc-test -o wide

# 查看服务
kubectl get svc -n mpc-test### 查看日志
ash
# Company 节点日志
kubectl logs -n mpc-test -l app.kubernetes.io/component=company -f

# Partner 节点日志
kubectl logs -n mpc-test -l app.kubernetes.io/component=partner -f

# WebUI 日志
kubectl logs -n mpc-test -l app.kubernetes.io/component=webui -f### 升级和回滚

# 升级 Chart
helm upgrade mpc-test ./helm-chart/mobile-mpc-project -n mpc-test -f values.yaml

# 查看历史版本
helm history mpc-test -n mpc-test

# 回滚到上一版本
helm rollback mpc-test -n mpc-test

# 回滚到指定版本
helm rollback mpc-test 1 -n mpc-test### 卸载

# 卸载单集群部署
helm uninstall mpc-test -n mpc-test

# 卸载多集群部署
kubectl config use-context minikube
helm uninstall mpc-company -n mpc-test

kubectl config use-context cluster-b
helm uninstall mpc-partner -n mpc-test
## 故障排查

### Pod 启动失败

# 查看 Pod 详情
kubectl describe pod <pod-name> -n mpc-test

# 查看事件
kubectl get events -n mpc-test --sort-by='.lastTimestamp'

# 查看 Pod 日志
kubectl logs <pod-name> -n mpc-test --previous### 网络连接问题

#### 单集群模式
h
# 测试 Service DNS 解析
kubectl run -it --rm debug --image=busybox --restart=Never -n mpc-test -- sh
nslookup mpc-test-mobile-mpc-project-company-svc
nslookup mpc-test-mobile-mpc-project-partner-svc#### 多集群模式

# 测试节点 IP 连通性（从 Company Pod）
kubectl exec -it <company-pod> -n mpc-test -- bash
ping 192.168.58.2  # 集群 B 节点 IP
nc -zv 192.168.58.2 9395  # 测试端口

# 测试节点 IP 连通性（从 Partner Pod）
kubectl exec -it <partner-pod> -n mpc-test -- bash
ping 192.168.49.2  # 集群 A 节点 IP
nc -zv 192.168.49.2 6379  # 测试 Ray Head 端口
### Ray 连接问题

#### 检查 Ray Head 状态

# 在 Company Pod 中
kubectl exec -it <company-pod> -n mpc-test -- bash
ray status#### 检查 Ray Worker 连接

# 在 Partner Pod 中查看日志
kubectl logs -n mpc-test -l app.kubernetes.io/component=partner -f

# 常见错误：
# - "GCS failed to check the health"：通常是网络地址配置问题
# - "Connection refused"：检查节点 IP 和端口是否正确### 镜像拉取问题

# 检查镜像是否存在
docker images | grep mobile-mpc

# 如果使用 Minikube，确保镜像已加载
minikube image ls | grep mobile-mpc

# 重新加载镜像
docker save mobile-mpc-company:v1.0.16 | minikube image load -### WebUI 无法访问

# 检查 WebUI Pod 状态
kubectl get pods -l app.kubernetes.io/component=webui -n mpc-test

# 检查 Service
kubectl get svc -l app.kubernetes.io/component=webui -n mpc-test

# 检查端口转发
kubectl port-forward -n mpc-test svc/mpc-test-mobile-mpc-project-webui 8080:8080

# 查看 WebUI 日志
kubectl logs -n mpc-test -l app.kubernetes.io/component=webui## 多集群部署注意事项

### 1. 节点 IP 配置

- **必须使用节点 IP**，不能使用 Pod IP 或 Service IP
- 确保两个集群的节点 IP 可以互相访问
- 如果节点 IP 变化，需要更新配置文件并重新部署

### 2. hostNetwork 模式

- 启用 `hostNetwork: true` 后，Pod 直接使用节点网络
- 端口冲突：确保节点上的端口（9394、9395、6379 等）未被占用
- 滚动更新时可能出现端口冲突，建议使用 `Recreate` 策略

### 3. 防火墙和网络策略

- 确保两个集群间的以下端口开放：
  - Company → Partner: `9395` (SPU)
  - Partner → Company: `9394` (SPU), `9396` (Coordinator), `6379` (Ray Head)

### 4. 镜像同步

- 确保两个集群都有对应的镜像
- 如果使用本地镜像，需要在每个集群的节点上加载镜像

### 5. 数据持久化

- 多集群部署时，每个集群的数据是独立的
- 模型文件保存在各自集群的持久化存储中
- 如果需要共享数据，需要配置共享存储（如 NFS）

