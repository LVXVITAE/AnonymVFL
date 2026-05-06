# 三机部署指南

本文档描述当前项目的推荐生产部署方式：将 Company、Partner、Coordinator 分别部署到三台机器或三个 Kubernetes 集群中，避免 Company 与 Coordinator 同机带来的安全风险。

## 目标拓扑

- Machine A / Cluster A: Company
  - Ray Head
  - Company SPU
  - Company WebUI（可写）
- Machine B / Cluster B: Partner
  - Ray Worker
  - Partner SPU
  - Partner WebUI（只读）
- Machine C / Cluster C: Coordinator
  - Ray Worker
  - Coordinator SPU

## 关键端口

| 角色 | 端口 | 说明 |
|------|------|------|
| Company | 6379 | Ray Head |
| Company | 9394 | Company SPU |
| Company WebUI | 30080 | NodePort 示例 |
| Partner | 9395 | Partner SPU |
| Partner WebUI | 30081 | NodePort 示例 |
| Coordinator | 9396 | Coordinator SPU |

如果使用 `hostNetwork: true`，Machine A/B/C 之间需要放通上述端口。

## 相关文件

- [build-multi-cluster-images.sh](build-multi-cluster-images.sh): 构建三类镜像
- [deploy-multi-cluster.sh](deploy-multi-cluster.sh): 三集群部署脚本
- [Dockerfile.company](Dockerfile.company): Company 镜像
- [Dockerfile.partner](Dockerfile.partner): Partner 镜像
- [Dockerfile.coordinator](Dockerfile.coordinator): Coordinator 镜像
- [helm-chart/mobile-mpc-project/values-cluster-a.yaml](helm-chart/mobile-mpc-project/values-cluster-a.yaml): Company 集群配置
- [helm-chart/mobile-mpc-project/values-cluster-b.yaml](helm-chart/mobile-mpc-project/values-cluster-b.yaml): Partner 集群配置
- [helm-chart/mobile-mpc-project/values-cluster-c.yaml](helm-chart/mobile-mpc-project/values-cluster-c.yaml): Coordinator 集群配置
- [web_ui/config.yaml](web_ui/config.yaml): WebUI 网络与节点配置示例
- [web_ui/test_network.py](web_ui/test_network.py): 三机连通性检查脚本

## 前置条件

1. 三台机器或三个 Kubernetes 集群之间网络互通。
2. Company 机器可访问 Partner `9395` 和 Coordinator `9396`。
3. Partner 与 Coordinator 可访问 Company `6379`。
4. 本地具备以下工具：
   - `docker`
   - `kubectl`
   - `helm`
5. 三个 kube context 已就绪：
   - `cluster-a`
   - `cluster-b`
   - `cluster-c`

## 第一步：确认三机地址

示例：

- Company: `192.168.10.11`
- Partner: `192.168.10.12`
- Coordinator: `192.168.10.13`

在部署前，请把这三个地址分别代入：

- [helm-chart/mobile-mpc-project/values-cluster-a.yaml](helm-chart/mobile-mpc-project/values-cluster-a.yaml)
- [helm-chart/mobile-mpc-project/values-cluster-b.yaml](helm-chart/mobile-mpc-project/values-cluster-b.yaml)
- [helm-chart/mobile-mpc-project/values-cluster-c.yaml](helm-chart/mobile-mpc-project/values-cluster-c.yaml)

如果使用部署脚本，则无需手工替换，脚本会自动完成占位符替换。

## 第二步：构建三类镜像

执行：

```bash
./build-multi-cluster-images.sh v1.0.19
```

脚本会构建并导出：

- `mobile-mpc-company:v1.0.19`
- `mobile-mpc-partner:v1.0.19`
- `mobile-mpc-coordinator:v1.0.19`

以及对应 tar 文件：

- `mobile-mpc-company-v1.0.19.tar`
- `mobile-mpc-partner-v1.0.19.tar`
- `mobile-mpc-coordinator-v1.0.19.tar`

## 第三步：加载镜像到目标环境

如果目标集群不直接拉取镜像仓库，可在对应节点或集群中加载 tar：

```bash
docker load -i mobile-mpc-company-v1.0.19.tar
docker load -i mobile-mpc-partner-v1.0.19.tar
docker load -i mobile-mpc-coordinator-v1.0.19.tar
```

## 第四步：校验 Helm 模板

建议在正式部署前执行：

```bash
helm lint helm-chart/mobile-mpc-project

helm template mpc-company helm-chart/mobile-mpc-project \
  -f helm-chart/mobile-mpc-project/values-cluster-a.yaml

helm template mpc-partner helm-chart/mobile-mpc-project \
  -f helm-chart/mobile-mpc-project/values-cluster-b.yaml

helm template mpc-coordinator helm-chart/mobile-mpc-project \
  -f helm-chart/mobile-mpc-project/values-cluster-c.yaml
```

重点检查：

1. Company Deployment 不再暴露 Coordinator 端口。
2. Coordinator Service selector 指向 `app.kubernetes.io/component: coordinator`。
3. Company Ray Head 只声明 `company` 资源。
4. Coordinator Deployment 声明 `coordinator` 资源并通过 `RAY_HEAD_ADDR` 加入 Company。

## 第五步：执行三集群部署

直接使用脚本：

```bash
export CLUSTER_A_CONTEXT=cluster-a
export CLUSTER_B_CONTEXT=cluster-b
export CLUSTER_C_CONTEXT=cluster-c

export CLUSTER_A_NODE_IP=192.168.10.11
export CLUSTER_B_NODE_IP=192.168.10.12
export CLUSTER_C_NODE_IP=192.168.10.13

./deploy-multi-cluster.sh
```

该脚本会：

1. 检查三个集群的连接状态。
2. 加载 Company、Partner、Coordinator 三类镜像。
3. 生成三份临时 values 文件并替换 IP 占位符。
4. 依次部署：
   - `mpc-company`
   - `mpc-partner`
   - `mpc-coordinator`

## 第六步：查看部署状态

```bash
kubectl --context=cluster-a -n mpc-test get pods -o wide
kubectl --context=cluster-b -n mpc-test get pods -o wide
kubectl --context=cluster-c -n mpc-test get pods -o wide
```

预期：

- Cluster A: `company`、`webui`
- Cluster B: `partner`、`webui`
- Cluster C: `coordinator`

查看日志：

```bash
kubectl --context=cluster-a -n mpc-test logs -l app.kubernetes.io/component=company -f
kubectl --context=cluster-b -n mpc-test logs -l app.kubernetes.io/component=partner -f
kubectl --context=cluster-c -n mpc-test logs -l app.kubernetes.io/component=coordinator -f
```

## 第七步：验证三机分离

### 方式一：检查 WebUI 配置

查看 [web_ui/config.yaml](web_ui/config.yaml)，确认：

- `company.ip` 不等于 `partner.ip`
- `partner.ip` 不等于 `coordinator.ip`
- `company.ip` 不等于 `coordinator.ip`

### 方式二：执行网络测试脚本

```bash
python web_ui/test_network.py
```

脚本会检查：

1. Company 的 Ray 端口和 SPU 端口。
2. Partner 的 SPU 端口。
3. Coordinator 的 SPU 端口。
4. 三个节点 IP 是否彼此不同。
5. 本地 WebUI 端口与可选远端 WebUI API 连通性。

### 方式三：检查 Company 容器是否仍承载 Coordinator

```bash
kubectl --context=cluster-a -n mpc-test describe pod -l app.kubernetes.io/component=company
kubectl --context=cluster-c -n mpc-test describe pod -l app.kubernetes.io/component=coordinator
```

检查点：

- Company 不应再监听 `9396`
- Coordinator 应独立监听 `9396`

## 第八步：发起训练与推理

- Company WebUI 地址：`http://<CLUSTER_A_NODE_IP>:30080`
- Partner WebUI 地址：`http://<CLUSTER_B_NODE_IP>:30081`

训练和推理由 Company WebUI 发起。
Partner WebUI 仅用于只读观察和数据集协同。

当前实现中，训练/推理命令会使用环境变量 `COORDINATOR_SPU_ADDR`，不会再把 Coordinator 地址自动绑定到 Company 本地 IP。

## 故障排查

### 1. Partner 或 Coordinator 无法加入 Ray

检查：

```bash
kubectl --context=cluster-a -n mpc-test logs -l app.kubernetes.io/component=company
kubectl --context=cluster-b -n mpc-test logs -l app.kubernetes.io/component=partner
kubectl --context=cluster-c -n mpc-test logs -l app.kubernetes.io/component=coordinator
```

确认：

- Company `6379` 可达
- `RAY_HEAD_ADDR` 配置正确
- 三台机器间无防火墙阻断

### 2. 训练报 coordinator 地址错误

检查 Company Deployment 环境变量：

```bash
kubectl --context=cluster-a -n mpc-test get deploy -o yaml | grep -n COORDINATOR_SPU_ADDR
```

应指向 Machine C，例如：

```text
COORDINATOR_SPU_ADDR=192.168.10.13:9396
```

### 3. WebUI 能打开但训练失败

重点检查：

- Partner `9395` 是否可达
- Coordinator `9396` 是否可达
- Company Pod 日志中输出的三方地址是否分别落在三台机器上

## 回滚与清理

卸载：

```bash
helm --kube-context=cluster-a uninstall mpc-company -n mpc-test
helm --kube-context=cluster-b uninstall mpc-partner -n mpc-test
helm --kube-context=cluster-c uninstall mpc-coordinator -n mpc-test
```

删除命名空间：

```bash
kubectl --context=cluster-a delete namespace mpc-test
kubectl --context=cluster-b delete namespace mpc-test
kubectl --context=cluster-c delete namespace mpc-test
```

## 当前实现范围

本次改造已经完成以下核心能力：

1. Coordinator 独立 Deployment 与镜像。
2. Company 不再兼任 Coordinator。
3. WebUI 训练与推理链路使用独立 Coordinator 地址。
4. 三集群部署脚本支持 A/B/C 三端。

未覆盖内容：

1. mTLS 或服务间双向认证。
2. 更严格的 NetworkPolicy。
3. Coordinator 最小权限镜像裁剪。

如果要进一步提升生产安全性，建议下一阶段加入网络策略、访问白名单和服务审计。
