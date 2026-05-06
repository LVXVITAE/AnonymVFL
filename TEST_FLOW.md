# 三机联调测试流程

本文档用于验证以下三机部署能力是否完整可用：

1. Company、Partner、Coordinator 分别运行在不同机器
2. Company WebUI 可以发起训练和推理
3. Partner WebUI 只读并可同步训练状态
4. Coordinator 只作为独立安全计算参与方，不再与 Company 同机

## 0. 目标拓扑

- Machine A: Company + Company WebUI
- Machine B: Partner + Partner WebUI
- Machine C: Coordinator

## 1. 前置准备

### 1.1 环境

- Docker
- kubectl
- Helm
- 三个 Kubernetes context
- 三台机器互通

### 1.2 关键配置

确认以下文件中的地址已对应三台机器：

- [helm-chart/mobile-mpc-project/values-cluster-a.yaml](helm-chart/mobile-mpc-project/values-cluster-a.yaml)
- [helm-chart/mobile-mpc-project/values-cluster-b.yaml](helm-chart/mobile-mpc-project/values-cluster-b.yaml)
- [helm-chart/mobile-mpc-project/values-cluster-c.yaml](helm-chart/mobile-mpc-project/values-cluster-c.yaml)
- [web_ui/config.yaml](web_ui/config.yaml)

## 2. 构建并加载镜像

```bash
./build-multi-cluster-images.sh v1.0.19
```

确认构建了三类镜像：

- Company
- Partner
- Coordinator

## 3. 部署三端

```bash
export CLUSTER_A_CONTEXT=cluster-a
export CLUSTER_B_CONTEXT=cluster-b
export CLUSTER_C_CONTEXT=cluster-c

export CLUSTER_A_NODE_IP=192.168.10.11
export CLUSTER_B_NODE_IP=192.168.10.12
export CLUSTER_C_NODE_IP=192.168.10.13

./deploy-multi-cluster.sh
```

## 4. 部署后检查

```bash
kubectl --context=${CLUSTER_A_CONTEXT} -n mpc-test get pods -o wide
kubectl --context=${CLUSTER_B_CONTEXT} -n mpc-test get pods -o wide
kubectl --context=${CLUSTER_C_CONTEXT} -n mpc-test get pods -o wide
```

预期：

- Cluster A: company、webui
- Cluster B: partner、webui
- Cluster C: coordinator

## 5. WebUI 验证

### 5.1 Company WebUI

访问：

```text
http://<CLUSTER_A_NODE_IP>:30080
```

检查：

- 可执行训练
- 可执行推理
- 可看到 Company 数据集
- 不应把 Coordinator 显示为 Company 本机角色

### 5.2 Partner WebUI

访问：

```text
http://<CLUSTER_B_NODE_IP>:30081
```

检查：

- 可查看状态
- 不具备训练写权限
- 能同步看到 Company 发起的训练状态

## 6. 网络验证

执行：

```bash
python web_ui/test_network.py
```

预期：

1. 三个节点 IP 不相同
2. Company `6379/9394` 可达
3. Partner `9395` 可达
4. Coordinator `9396` 可达
5. Company/Partner WebUI API 可达

## 7. 训练验证

在 Company WebUI 发起一次训练。

重点检查：

- Company 日志中输出的 `Partner SPU 地址` 指向 Machine B
- Company 日志中输出的 `Coordinator SPU 地址` 指向 Machine C
- Partner 状态同步正常
- 训练完成后模型才显示

建议同时看三端日志：

```bash
kubectl --context=${CLUSTER_A_CONTEXT} -n mpc-test logs -l app.kubernetes.io/component=company -f
kubectl --context=${CLUSTER_B_CONTEXT} -n mpc-test logs -l app.kubernetes.io/component=partner -f
kubectl --context=${CLUSTER_C_CONTEXT} -n mpc-test logs -l app.kubernetes.io/component=coordinator -f
```

## 8. 推理验证

在 Company WebUI 发起推理。

重点检查：

- `ray_head_addr` 指向 Machine A
- `partner_spu_addr` 指向 Machine B
- `coordinator_spu_addr` 指向 Machine C
- 返回结果正常

## 9. 安全边界检查

### 9.1 Company 不再兼任 Coordinator

检查 Company Pod：

```bash
kubectl --context=${CLUSTER_A_CONTEXT} -n mpc-test describe pod -l app.kubernetes.io/component=company
```

确认 Company 不再承载 Coordinator 端口和 Coordinator 资源。

### 9.2 Coordinator 独立存在

检查 Coordinator Pod：

```bash
kubectl --context=${CLUSTER_C_CONTEXT} -n mpc-test describe pod -l app.kubernetes.io/component=coordinator
```

确认其独立监听 `9396` 并作为独立 Ray Worker 存在。

## 10. 常见失败点

### 10.1 Coordinator 无法加入 Ray

检查 Machine C 到 Machine A 的 `6379` 是否可达。

### 10.2 训练命令仍把 coordinator 指向 Company

检查 Company Deployment 环境变量中 `COORDINATOR_SPU_ADDR` 的值。

### 10.3 WebUI 能打开但训练失败

通常优先排查：

1. Partner `9395` 不通
2. Coordinator `9396` 不通
3. Company `6379` 不通
4. 三份 values 文件中 IP 仍有旧值

## 11. 结论判定

如果以下条件同时满足，则说明三机改造验证通过：

1. 三端 Pod 分布在三台不同机器
2. Company 不再承载 Coordinator
3. 训练成功
4. 推理成功
5. Partner WebUI 状态同步正常
6. 网络检查脚本中的三机分离检查通过
