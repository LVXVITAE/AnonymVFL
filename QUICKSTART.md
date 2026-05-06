# 快速开始

本指南用于快速启动当前推荐部署方式：Company、Partner、Coordinator 三机部署。

如果你只想尽快落地，请按下面 5 步执行。

## 前置条件

- Docker 已安装
- kubectl 已配置
- Helm 3.x 已安装
- 三台机器或三个 Kubernetes 集群网络互通
- 已知三台机器 IP：
  - Company 机器
  - Partner 机器
  - Coordinator 机器

## 第 1 步：构建三类镜像

在项目根目录执行：

```bash
./build-multi-cluster-images.sh v1.0.19
```

产物：

- `mobile-mpc-company:v1.0.19`
- `mobile-mpc-partner:v1.0.19`
- `mobile-mpc-coordinator:v1.0.19`

以及三份 tar 文件。

## 第 2 步：准备三集群变量

```bash
export CLUSTER_A_CONTEXT=cluster-a
export CLUSTER_B_CONTEXT=cluster-b
export CLUSTER_C_CONTEXT=cluster-c

export CLUSTER_A_NODE_IP=192.168.10.11
export CLUSTER_B_NODE_IP=192.168.10.12
export CLUSTER_C_NODE_IP=192.168.10.13
```

说明：

- Cluster A: Company
- Cluster B: Partner
- Cluster C: Coordinator

## 第 3 步：执行部署

```bash
./deploy-multi-cluster.sh
```

脚本会自动：

1. 检查三个集群连接
2. 加载三类镜像
3. 替换三份 values 文件中的 IP 占位符
4. 部署三个 release：
   - `mpc-company`
   - `mpc-partner`
   - `mpc-coordinator`

## 第 4 步：检查状态

```bash
kubectl --context=${CLUSTER_A_CONTEXT} -n mpc-test get pods -o wide
kubectl --context=${CLUSTER_B_CONTEXT} -n mpc-test get pods -o wide
kubectl --context=${CLUSTER_C_CONTEXT} -n mpc-test get pods -o wide
```

预期：

- Cluster A: company、webui
- Cluster B: partner、webui
- Cluster C: coordinator

## 第 5 步：检查网络与打开 WebUI

运行网络测试：

```bash
python web_ui/test_network.py
```

访问地址：

- Company WebUI: `http://<CLUSTER_A_NODE_IP>:30080`
- Partner WebUI: `http://<CLUSTER_B_NODE_IP>:30081`

训练和推理由 Company WebUI 发起。
Partner WebUI 仅用于只读观察和数据集协同。

## 常用命令

```bash
# 查看 Company 日志
kubectl --context=${CLUSTER_A_CONTEXT} -n mpc-test logs -l app.kubernetes.io/component=company -f

# 查看 Partner 日志
kubectl --context=${CLUSTER_B_CONTEXT} -n mpc-test logs -l app.kubernetes.io/component=partner -f

# 查看 Coordinator 日志
kubectl --context=${CLUSTER_C_CONTEXT} -n mpc-test logs -l app.kubernetes.io/component=coordinator -f
```

## 常见问题

### 1. Company 和 Coordinator 仍在同一机器

检查：

- [helm-chart/mobile-mpc-project/values-cluster-a.yaml](helm-chart/mobile-mpc-project/values-cluster-a.yaml)
- [helm-chart/mobile-mpc-project/values-cluster-c.yaml](helm-chart/mobile-mpc-project/values-cluster-c.yaml)

确认 `CLUSTER_A_NODE_IP` 与 `CLUSTER_C_NODE_IP` 不同。

### 2. 训练时报 coordinator 地址错误

检查 Company 部署环境变量：

```bash
kubectl --context=${CLUSTER_A_CONTEXT} -n mpc-test get deploy -o yaml | grep COORDINATOR_SPU_ADDR
```

### 3. 网络测试失败

如果当前集群尚未部署完成，`web_ui/test_network.py` 会显示端口不通，这是正常的。
真正需要关注的是：

- 三个节点 IP 是否不同
- 部署完成后端口是否打通

## 更多文档

- 详细部署： [DEPLOYMENT.md](DEPLOYMENT.md)
- Chart 文档： [helm-chart/mobile-mpc-project/README.md](helm-chart/mobile-mpc-project/README.md)
- 本地推理复现： [LOCAL_INFER_REPRO.md](LOCAL_INFER_REPRO.md)
- 联调流程： [TEST_FLOW.md](TEST_FLOW.md)
