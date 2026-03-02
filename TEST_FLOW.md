# Mobile MPC 双集群联调测试流程（Company + Partner）

## 0. 目标

验证以下功能在新机器可完整复现：

1. Company WebUI 不显示 Partner 数据集输入框
2. 训练/推理可自动使用 Partner 端已上传数据集（无上传时回退默认路径）
3. Partner 只读 WebUI 可同步显示训练状态
4. 模型仅在训练完成后显示（避免初始预置模型误显示）

---

## 1. 前置准备

### 1.1 代码目录

使用目录：`mobile_project3_new`  
不要使用旧目录：`260226`

### 1.2 环境

- Docker
- Minikube（双 profile：`minikube` + `cluster-b`）
- kubectl
- Helm
- 建议两个终端窗口分别操作 A/B 集群

### 1.3 关键配置文件确认

- `helm-chart/mobile-mpc-project/values-cluster-a.yaml`
  - `company.image.tag: v1.0.19`
  - `webui.image.tag: v1.0.19`
  - `webui.env` 包含 `PARTNER_WEBUI_URL=http://<cluster-b-node-ip>:30081`
- `helm-chart/mobile-mpc-project/values-cluster-b.yaml`
  - `partner.image.tag: v1.0.19`
  - `webui.image.tag: v1.0.19`
  - `webui.env` 包含 `COMPANY_WEBUI_URL=http://<cluster-a-node-ip>:30080`

---

## 2. 构建并加载镜像

在 `mobile_project3_new` 目录执行：

```bash
sudo docker build -t mobile-mpc-company:v1.0.19 -f Dockerfile.company .
sudo docker build -t mobile-mpc-partner:v1.0.19 -f Dockerfile.partner .

sudo minikube image load mobile-mpc-company:v1.0.19
sudo minikube image load mobile-mpc-partner:v1.0.19 -p cluster-b
```

可选检查：

```bash
sudo minikube image ls | rg "mobile-mpc-company|mobile-mpc-partner"
sudo minikube image ls -p cluster-b | rg "mobile-mpc-company|mobile-mpc-partner"
```

---

## 3. 部署双集群

### 3.1 部署 Company（cluster-a / minikube）

```bash
sudo helm upgrade --install mobile-mpc ./helm-chart/mobile-mpc-project \
  -f ./helm-chart/mobile-mpc-project/values-cluster-a.yaml \
  --namespace mpc --create-namespace
```

### 3.2 部署 Partner（cluster-b）

```bash
sudo helm upgrade --install mobile-mpc ./helm-chart/mobile-mpc-project \
  -f ./helm-chart/mobile-mpc-project/values-cluster-b.yaml \
  --namespace mpc --create-namespace --kube-context cluster-b
```

### 3.3 检查 Pod

```bash
sudo kubectl get pods -n mpc
sudo kubectl --context cluster-b get pods -n mpc
```

---

## 4. 打开 WebUI

### 4.1 Company WebUI

```bash
sudo kubectl port-forward -n mpc svc/mobile-mpc-webui 8080:8080
```

浏览器访问：`http://localhost:8080`

### 4.2 Partner WebUI（新终端）

```bash
sudo kubectl --context cluster-b port-forward -n mpc svc/mobile-mpc-webui 8081:8080
```

浏览器访问：`http://localhost:8081`

---

## 5. 功能验证步骤

### 5.1 UI 显示验证

- Company WebUI：
  - 训练区只显示 Company 数据集下拉框
  - 推理区不显示 Partner 推理数据集输入框（仅说明自动获取）
- Partner WebUI：
  - 只显示 Partner 自身数据集上传/选择区域（只读控制下不应触发训练）

### 5.2 数据集上传验证

- 在 Partner WebUI 上传一个新 CSV（建议文件名包含 `train`/`test` 或 `val` 关键词）
- 在 Company WebUI 上传 Company 数据集

### 5.3 训练验证

- 仅在 Company WebUI 点击“开始训练”
- 观察：
  - Company 训练状态变化（运行中 -> 已完成）
  - Partner WebUI 状态同步变化（通过代理状态接口）
  - 训练完成后模型列表出现新模型（非启动即显示）

### 5.4 推理验证

- 在 Company WebUI 启动推理
- 预期：
  - 后端自动解析 Partner 推理数据路径（优先 Partner 上传 test/val；否则回退默认）
  - 推理成功返回结果

---

## 6. 日志排查命令

### Company 集群

```bash
sudo kubectl get pods -n mpc
sudo kubectl logs -n mpc <company-pod-name> --tail=200
sudo kubectl logs -n mpc <webui-pod-name> --tail=200
```

### Partner 集群

```bash
sudo kubectl --context cluster-b get pods -n mpc
sudo kubectl --context cluster-b logs -n mpc <partner-pod-name> --tail=200
sudo kubectl --context cluster-b logs -n mpc <webui-pod-name> --tail=200
```

---

## 7. 关键注意事项

1. 不要混用旧目录  
   当前有效目录是 `mobile_project3_new`，`260226` 为历史目录。

2. IP/URL 必须匹配实际节点  
   `PARTNER_WEBUI_URL`、`COMPANY_WEBUI_URL` 需要用真实 NodeIP + NodePort。

3. 镜像 tag 必须一致  
   构建 tag 与 values 文件 tag 必须完全一致（都用 `v1.0.19`）。

4. 双集群 context 不要搞混  
   cluster-a 默认 `minikube`，cluster-b 使用 `--context cluster-b`。

5. 端口转发需保持前台运行  
   8080/8081 的 `port-forward` 终端不能关闭。

6. 全匿踪边界说明  
   Company 获取的是 Partner 数据集路径/元信息，不是 CSV 明文；Partner 数据在其本地读取并参与安全计算。
