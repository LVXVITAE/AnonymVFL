# 快速入门指南

本指南帮助你在 5 分钟内将项目部署到 Kubernetes。

## 前置条件

- ✅ Docker 已安装
- ✅ Kubernetes 集群可访问（kubectl 已配置）
- ✅ Helm 3.x 已安装
- ✅ 镜像仓库访问权限

## 三步部署

### 第一步：构建并推送镜像

```bash
# 进入项目目录
cd /home/dxn/mobile_project_final/mobile_project3_new

# 修改为你的镜像仓库地址
export REGISTRY="registry.example.com/mpc"
export IMAGE_TAG="v1.0.0"

# 构建镜像
docker build -t ${REGISTRY}/mobile-mpc-project:${IMAGE_TAG} .

# 推送镜像
docker push ${REGISTRY}/mobile-mpc-project:${IMAGE_TAG}
```

### 第二步：配置 Chart

创建自定义配置文件 `my-values.yaml`：

```yaml
global:
  imageRegistry: "registry.example.com/mpc/"  # 修改为你的仓库

company:
  image:
    tag: "v1.0.0"

partner:
  image:
    tag: "v1.0.0"

webui:
  service:
    type: NodePort  # 或 LoadBalancer
```

### 第三步：部署到 Kubernetes

```bash
# 创建命名空间
kubectl create namespace mpc-project

# 如果使用私有仓库，创建镜像拉取密钥
kubectl create secret docker-registry regcred \
  --docker-server=registry.example.com \
  --docker-username=YOUR_USERNAME \
  --docker-password=YOUR_PASSWORD \
  -n mpc-project

# 安装 Chart
helm install my-mpc helm-chart/mobile-mpc-project \
  -n mpc-project \
  -f my-values.yaml

# 查看状态
kubectl get pods -n mpc-project -w
```

## 验证部署

```bash
# 查看所有资源
kubectl get all -n mpc-project

# 查看 Pod 日志
kubectl logs -n mpc-project -l app.kubernetes.io/component=company -f

# 访问 Web UI（如果使用 NodePort）
kubectl get svc -n mpc-project
# 访问 http://<NODE_IP>:<NODE_PORT>
```

## 常用命令

```bash
# 查看 Helm 发布
helm list -n mpc-project

# 升级应用
helm upgrade my-mpc helm-chart/mobile-mpc-project -n mpc-project

# 卸载应用
helm uninstall my-mpc -n mpc-project

# 查看详细信息
helm status my-mpc -n mpc-project
```

## 故障排查

### Pod 启动失败

```bash
kubectl describe pod <pod-name> -n mpc-project
kubectl logs <pod-name> -n mpc-project
```

### 镜像拉取失败

检查镜像是否存在：
```bash
docker pull ${REGISTRY}/mobile-mpc-project:${IMAGE_TAG}
```

确认镜像拉取密钥配置正确。

## 下一步

- 📖 详细文档：查看 [DEPLOYMENT.md](./DEPLOYMENT.md)
- 📋 Chart 文档：查看 [helm-chart/mobile-mpc-project/README.md](./helm-chart/mobile-mpc-project/README.md)
- ⚙️ 配置说明：查看 [values.yaml](./helm-chart/mobile-mpc-project/values.yaml)

## 参考

### 目录结构

```
mobile_project3_new/
├── Dockerfile                  # Docker 镜像构建文件
├── requirements.txt            # Python 依赖
├── .dockerignore              # Docker 构建忽略
├── QUICKSTART.md              # 本文件
├── DEPLOYMENT.md              # 详细部署文档
├── validate-chart.sh          # Chart 验证脚本
└── helm-chart/                # Helm Chart
    └── mobile-mpc-project/
        ├── Chart.yaml
        ├── values.yaml
        └── templates/
```

### 已创建的 Kubernetes 资源

- ✅ Company Deployment（甲方节点）
- ✅ Partner Deployment（乙方节点）
- ✅ Services（网络服务）
- ✅ ServiceAccount（服务账户）
- ✅ Ingress（入口，可选）
- ✅ PVC（持久化存储，可选）

### 示例：本地 Minikube 部署

如果你在本地使用 Minikube：

```bash
# 启动 Minikube
minikube start

# 使用 Minikube 的 Docker 环境
eval $(minikube docker-env)

# 构建镜像（直接在 Minikube 中）
docker build -t mobile-mpc-project:v1.0.0 .

# 部署（不需要推送到远程仓库）
helm install my-mpc helm-chart/mobile-mpc-project \
  -n mpc-project \
  --create-namespace \
  --set global.imageRegistry="" \
  --set company.image.repository=mobile-mpc-project \
  --set company.image.tag=v1.0.0 \
  --set company.image.pullPolicy=Never \
  --set partner.image.repository=mobile-mpc-project \
  --set partner.image.tag=v1.0.0 \
  --set partner.image.pullPolicy=Never

# 访问服务
minikube service my-mpc-mobile-mpc-project-webui -n mpc-project
```

## 需要帮助？

如有问题，请查看：
1. Pod 日志：`kubectl logs -n mpc-project <pod-name>`
2. 事件：`kubectl get events -n mpc-project`
3. 详细文档：`DEPLOYMENT.md`

