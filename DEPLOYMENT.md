# 部署指南 - 移动多方安全计算项目

本文档详细说明如何将项目打包成 Helm Chart 并部署到 Kubernetes 集群。

## 目录结构

```
mobile_project3_new/
├── company/                    # 甲方代码
├── partner/                    # 乙方代码
├── web_ui/                     # Web 界面
├── trans/                      # 传输模块
├── test/                       # 测试文件
├── Dockerfile                  # Docker 镜像构建文件
├── requirements.txt            # Python 依赖
├── .dockerignore              # Docker 构建忽略文件
└── helm-chart/                 # Helm Chart 目录
    └── mobile-mpc-project/
        ├── Chart.yaml          # Chart 元数据
        ├── values.yaml         # 默认配置值
        ├── README.md           # Chart 文档
        ├── .helmignore        # Helm 打包忽略文件
        └── templates/          # Kubernetes 资源模板
            ├── _helpers.tpl
            ├── company-deployment.yaml
            ├── partner-deployment.yaml
            ├── service.yaml
            ├── serviceaccount.yaml
            ├── ingress.yaml
            ├── pvc.yaml
            └── NOTES.txt
```

## 部署步骤

### 第一步：构建 Docker 镜像

1. **准备镜像仓库**

   确保你有可访问的 Docker 镜像仓库（Docker Hub、Harbor、阿里云等）。

2. **登录镜像仓库**

   ```bash
   # Docker Hub
   docker login
   
   # 私有仓库
   docker login registry.example.com
   ```

3. **构建镜像**

   ```bash
   cd /home/dxn/mobile_project_final/mobile_project3_new
   
   # 构建镜像（替换为你的仓库地址）
   docker build -t registry.example.com/mpc/mobile-mpc-project:v1.0.0 .
   
   # 也可以打多个标签
   docker tag registry.example.com/mpc/mobile-mpc-project:v1.0.0 \
              registry.example.com/mpc/mobile-mpc-project:latest
   ```

4. **推送镜像**

   ```bash
   docker push registry.example.com/mpc/mobile-mpc-project:v1.0.0
   docker push registry.example.com/mpc/mobile-mpc-project:latest
   ```

5. **验证镜像**

   ```bash
   # 测试镜像是否可以正常运行
   docker run --rm registry.example.com/mpc/mobile-mpc-project:v1.0.0 python --version
   ```

### 第二步：配置 Helm Chart

1. **编辑 values.yaml**

   根据你的环境修改 `helm-chart/mobile-mpc-project/values.yaml`：

   ```yaml
   global:
     # 修改为你的镜像仓库地址
     imageRegistry: "registry.example.com/mpc/"
     imagePullPolicy: IfNotPresent
   
   company:
     enabled: true
     image:
       repository: mobile-mpc-project
       tag: "v1.0.0"
     resources:
       limits:
         cpu: "2"
         memory: 4Gi
       requests:
         cpu: "1"
         memory: 2Gi
   
   partner:
     enabled: true
     image:
       repository: mobile-mpc-project
       tag: "v1.0.0"
   
   webui:
     enabled: true
     service:
       type: LoadBalancer  # 或 NodePort、ClusterIP
   ```

2. **创建自定义配置文件（可选）**

   ```bash
   cat > custom-values.yaml <<EOF
   global:
     imageRegistry: "your-registry.com/mpc/"
   
   company:
     resources:
       limits:
         cpu: "4"
         memory: 8Gi
   
   webui:
     ingress:
       enabled: true
       hosts:
         - host: mpc.example.com
           paths:
             - path: /
               pathType: Prefix
   EOF
   ```

### 第三步：验证 Helm Chart

1. **检查 Chart 语法**

   ```bash
   cd /home/dxn/mobile_project_final/mobile_project3_new
   
   # Lint 检查
   helm lint helm-chart/mobile-mpc-project
   ```

2. **模板渲染测试**

   ```bash
   # 渲染模板查看生成的 YAML
   helm template my-mpc helm-chart/mobile-mpc-project
   
   # 使用自定义配置渲染
   helm template my-mpc helm-chart/mobile-mpc-project -f custom-values.yaml
   
   # 输出到文件检查
   helm template my-mpc helm-chart/mobile-mpc-project > rendered.yaml
   ```

3. **Dry-run 测试**

   ```bash
   # 模拟安装（不实际创建资源）
   helm install my-mpc helm-chart/mobile-mpc-project --dry-run --debug
   ```

### 第四步：部署到 Kubernetes

1. **确保 Kubernetes 集群可访问**

   ```bash
   kubectl cluster-info
   kubectl get nodes
   ```

2. **创建命名空间**

   ```bash
   kubectl create namespace mpc-project
   ```

3. **创建镜像拉取密钥（如果使用私有仓库）**

   ```bash
   kubectl create secret docker-registry regcred \
     --docker-server=registry.example.com \
     --docker-username=<your-username> \
     --docker-password=<your-password> \
     --docker-email=<your-email> \
     -n mpc-project
   ```
   
   然后在 values.yaml 中启用：
   ```yaml
   imagePullSecrets:
     - name: regcred
   ```

4. **安装 Chart**

   ```bash
   # 基本安装
   helm install my-mpc helm-chart/mobile-mpc-project -n mpc-project
   
   # 使用自定义配置安装
   helm install my-mpc helm-chart/mobile-mpc-project \
     -n mpc-project \
     -f custom-values.yaml
   
   # 安装并等待就绪
   helm install my-mpc helm-chart/mobile-mpc-project \
     -n mpc-project \
     --wait --timeout 10m
   ```

5. **查看部署状态**

   ```bash
   # 查看 Helm 发布
   helm list -n mpc-project
   
   # 查看发布详情
   helm status my-mpc -n mpc-project
   
   # 查看 Pod 状态
   kubectl get pods -n mpc-project
   
   # 查看所有资源
   kubectl get all -n mpc-project
   ```

### 第五步：访问应用

1. **查看服务**

   ```bash
   kubectl get svc -n mpc-project
   ```

2. **访问 Web UI**

   根据服务类型：

   **LoadBalancer:**
   ```bash
   export SERVICE_IP=$(kubectl get svc my-mpc-webui -n mpc-project \
     -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
   echo "访问地址: http://$SERVICE_IP:8080"
   ```

   **NodePort:**
   ```bash
   export NODE_IP=$(kubectl get nodes -o jsonpath='{.items[0].status.addresses[?(@.type=="ExternalIP")].address}')
   export NODE_PORT=$(kubectl get svc my-mpc-webui -n mpc-project \
     -o jsonpath='{.spec.ports[0].nodePort}')
   echo "访问地址: http://$NODE_IP:$NODE_PORT"
   ```

   **ClusterIP (端口转发):**
   ```bash
   kubectl port-forward -n mpc-project svc/my-mpc-webui 8080:8080
   # 访问 http://localhost:8080
   ```

3. **查看日志**

   ```bash
   # Company 节点日志
   kubectl logs -n mpc-project -l app.kubernetes.io/component=company -f
   
   # Partner 节点日志
   kubectl logs -n mpc-project -l app.kubernetes.io/component=partner -f
   
   # 所有 Pod 日志
   kubectl logs -n mpc-project --all-containers=true -l app.kubernetes.io/instance=my-mpc
   ```

### 第六步：打包 Chart（可选）

如果需要分发 Chart：

```bash
# 打包成 .tgz 文件
helm package helm-chart/mobile-mpc-project

# 生成 mobile-mpc-project-0.1.0.tgz

# 创建 Chart 仓库索引
helm repo index . --url https://your-charts-repo.com

# 上传到 Chart 仓库
# 可以使用 ChartMuseum、Harbor、GitHub Pages 等
```

## 升级和维护

### 升级应用

```bash
# 修改 values.yaml 或代码后

# 1. 重新构建镜像
docker build -t registry.example.com/mpc/mobile-mpc-project:v1.1.0 .
docker push registry.example.com/mpc/mobile-mpc-project:v1.1.0

# 2. 升级 Helm 发布
helm upgrade my-mpc helm-chart/mobile-mpc-project \
  -n mpc-project \
  --set company.image.tag=v1.1.0 \
  --set partner.image.tag=v1.1.0

# 或使用新的 values 文件
helm upgrade my-mpc helm-chart/mobile-mpc-project \
  -n mpc-project \
  -f custom-values.yaml
```

### 回滚

```bash
# 查看历史版本
helm history my-mpc -n mpc-project

# 回滚到上一版本
helm rollback my-mpc -n mpc-project

# 回滚到指定版本
helm rollback my-mpc 1 -n mpc-project
```

### 卸载

```bash
# 卸载 Helm 发布
helm uninstall my-mpc -n mpc-project

# 删除命名空间（慎用）
kubectl delete namespace mpc-project
```

## 故障排查

### Pod 启动失败

```bash
# 查看 Pod 详情
kubectl describe pod <pod-name> -n mpc-project

# 查看事件
kubectl get events -n mpc-project --sort-by='.lastTimestamp'

# 查看 Pod 日志
kubectl logs <pod-name> -n mpc-project

# 进入 Pod 调试
kubectl exec -it <pod-name> -n mpc-project -- /bin/bash
```

### 镜像拉取失败

```bash
# 检查镜像拉取密钥
kubectl get secret regcred -n mpc-project

# 测试镜像是否可访问
docker pull registry.example.com/mpc/mobile-mpc-project:v1.0.0

# 检查 Pod 的镜像拉取状态
kubectl describe pod <pod-name> -n mpc-project | grep -A 5 "Events"
```

### 服务连通性问题

```bash
# 测试服务 DNS 解析
kubectl run -it --rm debug --image=busybox --restart=Never -n mpc-project -- nslookup my-mpc-company-svc

# 测试端口连通性
kubectl run -it --rm debug --image=nicolaka/netshoot --restart=Never -n mpc-project -- bash
# 在容器内执行
curl my-mpc-company-svc:9394
telnet my-mpc-company-svc 9394
```

### 资源不足

```bash
# 查看节点资源使用情况
kubectl top nodes

# 查看 Pod 资源使用情况
kubectl top pods -n mpc-project

# 调整资源限制
helm upgrade my-mpc helm-chart/mobile-mpc-project \
  -n mpc-project \
  --set company.resources.limits.memory=8Gi
```

## 生产环境建议

1. **使用持久化存储**
   - 启用 PVC 保存模型和数据
   - 配置适当的 StorageClass

2. **配置资源限制**
   - 根据实际负载调整 CPU 和内存
   - 启用 HPA（水平自动扩缩容）

3. **安全配置**
   - 使用 RBAC 控制权限
   - 配置 Network Policy
   - 启用 Pod Security Policy

4. **监控和日志**
   - 集成 Prometheus 监控
   - 配置 ELK/EFK 日志收集
   - 设置告警规则

5. **高可用**
   - 配置多副本
   - 使用亲和性规则分散 Pod
   - 配置 PodDisruptionBudget

6. **备份策略**
   - 定期备份持久化数据
   - 保存 Helm values 配置
   - 记录镜像版本

## 参考资源

- [Helm 官方文档](https://helm.sh/docs/)
- [Kubernetes 官方文档](https://kubernetes.io/docs/)
- [SecretFlow 文档](https://www.secretflow.org.cn/)
- 项目 README: `helm-chart/mobile-mpc-project/README.md`

## 联系支持

如有问题，请联系技术团队：team@example.com

