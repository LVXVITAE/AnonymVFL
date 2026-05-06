# Mobile MPC Project Helm Chart

这个 Helm Chart 用于部署当前项目的三方安全计算系统，推荐部署形态为三机或三集群：

- Company
- Partner
- Coordinator

## 设计目标

当前版本的重点是避免 Company 与 Coordinator 同机部署，降低安全风险。

为此，Chart 现在支持：

- Company 独立 Deployment
- Partner 独立 Deployment
- Coordinator 独立 Deployment
- Company 可写 WebUI
- Partner 只读 WebUI

## 推荐部署模式

### 三集群模式

- Cluster A: Company + WebUI
- Cluster B: Partner + WebUI
- Cluster C: Coordinator

这是当前推荐模式。

### 单集群三节点模式

如果你的 Kubernetes 只有一个集群，也可以把 Company、Partner、Coordinator 分别调度到三个不同节点。此时需要结合 `nodeSelector` 或 `affinity` 使用。

## 镜像

需要三类镜像：

- `mobile-mpc-company:<tag>`
- `mobile-mpc-partner:<tag>`
- `mobile-mpc-coordinator:<tag>`

构建命令：

```bash
./build-multi-cluster-images.sh v1.0.19
```

## 关键 Values 文件

- [values.yaml](values.yaml): 默认值
- [values-cluster-a.yaml](values-cluster-a.yaml): Company 集群
- [values-cluster-b.yaml](values-cluster-b.yaml): Partner 集群
- [values-cluster-c.yaml](values-cluster-c.yaml): Coordinator 集群

## 三集群快速部署

### 1. 设置变量

```bash
export CLUSTER_A_CONTEXT=cluster-a
export CLUSTER_B_CONTEXT=cluster-b
export CLUSTER_C_CONTEXT=cluster-c

export CLUSTER_A_NODE_IP=192.168.10.11
export CLUSTER_B_NODE_IP=192.168.10.12
export CLUSTER_C_NODE_IP=192.168.10.13
```

### 2. 执行部署脚本

```bash
./deploy-multi-cluster.sh
```

### 3. 检查结果

```bash
kubectl --context=${CLUSTER_A_CONTEXT} -n mpc-test get pods
kubectl --context=${CLUSTER_B_CONTEXT} -n mpc-test get pods
kubectl --context=${CLUSTER_C_CONTEXT} -n mpc-test get pods
```

## 重要行为变化

与旧版本相比，当前 Chart 有以下关键变化：

1. Company 不再承载 Coordinator 端口。
2. Coordinator Service 不再指向 Company Pod。
3. Company Ray Head 资源不再包含 `coordinator`。
4. Coordinator 作为独立 Ray Worker 加入 Company Ray Head。

## 关键端口

| 组件 | 端口 | 说明 |
|------|------|------|
| Company | 6379 | Ray Head |
| Company | 9394 | Company SPU |
| Partner | 9395 | Partner SPU |
| Coordinator | 9396 | Coordinator SPU |
| Company WebUI | 30080 | 示例 NodePort |
| Partner WebUI | 30081 | 示例 NodePort |

## 验证建议

### Helm 检查

```bash
helm lint ./helm-chart/mobile-mpc-project
```

### 模板渲染

```bash
helm template mpc-company ./helm-chart/mobile-mpc-project -f ./helm-chart/mobile-mpc-project/values-cluster-a.yaml
helm template mpc-partner ./helm-chart/mobile-mpc-project -f ./helm-chart/mobile-mpc-project/values-cluster-b.yaml
helm template mpc-coordinator ./helm-chart/mobile-mpc-project -f ./helm-chart/mobile-mpc-project/values-cluster-c.yaml
```

### 网络检查

```bash
python web_ui/test_network.py
```

## 常见问题

### Coordinator 仍然落在 Company

检查：

- [templates/service.yaml](templates/service.yaml)
- [templates/coordinator-deployment.yaml](templates/coordinator-deployment.yaml)
- [values-cluster-c.yaml](values-cluster-c.yaml)

### Company 训练时 coordinator 地址仍是本机

检查：

- Company Deployment 环境变量 `COORDINATOR_SPU_ADDR`
- [web_ui/app.py](../../../web_ui/app.py) 中训练命令生成逻辑

## 相关文档

- 项目级部署文档： [../../DEPLOYMENT.md](../../DEPLOYMENT.md)
- 快速开始： [../../QUICKSTART.md](../../QUICKSTART.md)
- 联调流程： [../../TEST_FLOW.md](../../TEST_FLOW.md)
