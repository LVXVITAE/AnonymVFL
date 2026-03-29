# 联邦学习 Web UI 与算法 Python 接口文档

## 1. 项目概述

本项目为基于 SecretFlow / Ray 的纵向联邦学习（Company / Partner 双节点 + Coordinator）。接口分为两层：

| 层次 | 说明 | 主要源码 |
|------|------|----------|
| **控制台 HTTP/WebSocket** | 节点启停、训练编排、状态与日志查询 | `web_ui/app.py` |
| **联邦算法 Python API** | 纵向逻辑回归 `SSLR`、纵向 XGBoost `SSXGBoost`、基类 `SSML`、推理编排 `InferEngine` | `company/common.py`、`company/LR.py`、`company/XGBoost.py`、`company/infer.py` |

训练入口脚本为 `company/truerun.py`（由 Web UI 或命令行调用），不在本文逐行展开。

文档结构对齐《后量子签名-接口文档》：每项给出 **功能、参数、返回值、注意事项**。SecretFlow 类型（`SPU`、`PYU`、`SPUObject`、`PYUObject`、`FedNdarray` 等）以源码为准。

**基础约定（Web UI）**

- 默认 **Content-Type**：`application/json`（除 GET 无请求体）。
- **Base URL**：本地多为 `http://<host>:5000` 或 `8080`；Kubernetes 见 `helm-chart/mobile-mpc-project/values.yaml` 中 `webui.service.port`。
- 需 **Kubernetes API** 的接口仅在集群内且 `kubernetes` 客户端与 RBAC 可用时完整生效；否则部分能力降级为读本地日志文件。

---

## 2. Web UI HTTP 接口

### 2.1 服务说明

| 项 | 说明 |
|----|------|
| 框架 | Flask + Flask-SocketIO（`async_mode='threading'`） |
| 页面入口 | `GET /` → 渲染 `templates/index.html` |
| 跨域 | SocketIO 允许来源 `*` |

---

### 2.2 `GET /api/status`

**功能**：查询联邦学习控制台聚合状态（Company/Partner/训练状态、步数、准确率/损失占位、系统信息、最近日志摘要）。在 K8s 环境且客户端可用时，会按 Deployment 就绪情况刷新 Company/Partner 状态字符串。

**参数**：无（Query 无必填项）。

**返回值**：JSON，主要字段示例含义如下。

| 字段 | 类型 | 说明 |
|------|------|------|
| `company_status` | string | 如：`未启动`、`运行中`、`启动中`、`已停止` |
| `partner_status` | string | 同上 |
| `training_status` | string | 如：`未开始`、`PSI中`、`训练中`、`已完成` |
| `current_step` | number | 当前训练步（内存/解析结果） |
| `total_steps` | number | 总步数（占位） |
| `accuracy` | number | 准确率（占位/缓存） |
| `loss` | number | 损失（占位/缓存） |
| `system_info` | object | `cpu_percent`、`memory_percent`、`disk_percent`、`timestamp` |
| `logs` | array | 最近最多 50 条 `{timestamp, level, message}` |

**注意事项**：K8s 查询失败时不抛错给客户端，仍返回内存中的状态。

---

### 2.3 `GET /api/logs/<node_type>`

**功能**：获取指定节点日志行列表。优先通过 Kubernetes API 读取对应 Pod 日志；失败或未安装客户端时回退读本地文件。

**参数**（路径）：

| 参数 | 类型 | 说明 |
|------|------|------|
| `node_type` | string | `company` 或 `partner` |

**返回值**：JSON。

| 字段 | 类型 | 说明 |
|------|------|------|
| `status` | string | `success` 或 `error` |
| `logs` | array of string | 成功时最多约 100 行（尾部） |
| `message` | string | `error` 时错误说明 |

**回退文件路径**：

- `company` → `company/company_run.log`
- `partner` → `partner/partner_run.log`

**注意事项**：Pod 处于 `ContainerCreating` 等中间状态时可能返回空 `logs` 而 `status` 仍为 `success`。

---

### 2.4 `GET /api/training_metrics`

**功能**：从 Company 侧日志（K8s Pod 或 `company/company_run.log`）解析训练指标：当前 epoch/step、Accuracy/F1/FOR、最终指标及推导的训练状态字符串。

**参数**：无。

**返回值**：JSON，主要字段包括 `current_epoch`、`total_epochs`、`current_step`、`accuracy`、`f1`、`for`、`final_accuracy`、`final_f1`、`final_for`、`status` 等；解析异常时可能返回 `error`、`status: error`。

**注意事项**：依赖日志格式（如 `Step n, Accuracy: …, F1: …, FOR: …`、`最终准确率` 等正则）；状态与内存中的 `state.training_status` 会联动更新。

---

### 2.5 `GET /api/config` / `POST /api/config`

**功能**：读取或更新 Web UI 使用的 YAML 配置（内存 `state.config` 与 `web_ui/config.yaml` 持久化）。

**GET 参数**：无。

**GET 返回值**：完整配置对象（结构见 `SystemState.load_config` 默认项：`company`、`partner`、`coordinator`、`training` 等）。

**POST 参数**：JSON 对象，与现有配置 **深度合并**（`state.config.update(new_config)`），未传字段保持原值。

**POST 返回值**：`{"status": "success"}`。

**注意事项**：POST 会覆盖磁盘上的 `config.yaml`；训练相关默认值如 `training.model`（`SSLR` / `SSXGBoost`）、`default_epochs` 等由此维护。

---

### 2.6 `POST /api/start_company`

**功能**：将标签 `app.kubernetes.io/component=company` 的 Deployment 副本数设为 1，启动 Company 节点。

**参数**：无请求体（可不传 JSON）。

**返回值**：

| 情况 |  body 示例 |
|------|------------|
| 成功 | `{"status":"success","message":"Company节点启动中"}` 或已在运行时的提示 |
| 失败 | `{"status":"error","message":"..."}` |

**注意事项**：需要 `K8S_AVAILABLE`；命名空间来自环境变量 `NAMESPACE` 或 `POD_NAMESPACE`，默认 `mpc-test`。

---

### 2.7 `POST /api/start_partner`

**功能**：将标签 `app.kubernetes.io/component=partner` 的 Deployment 副本数设为 1，启动 Partner 节点。

**参数**：无请求体。

**返回值**：同 2.6，消息文案为 Partner。

**注意事项**：同 2.6，标签为 `partner`。

---

### 2.8 `POST /api/stop_company`

**功能**：将 Company Deployment 副本数设为 0。

**参数**：无请求体。

**返回值**：`status` + `message`；成功时可能通过 SocketIO 推送 `status_update`。

**注意事项**：停止后 Web UI 内存状态会更新为 `已停止`。

---

### 2.9 `POST /api/stop_partner`

**功能**：将 Partner Deployment 副本数设为 0。

**参数**、**返回值**、**注意事项**：同 2.8，对象为 Partner。

---

### 2.10 `POST /api/start_training`

**功能**：在 Company Deployment 已运行（副本数非 0）的前提下，改写容器 `args` 注入 `python company/truerun.py ...` 训练命令，并通过缩容至 0 再扩容至 1 触发 Pod 重启以执行训练。

**参数**：JSON，均可选；缺省取自 `state.config['training']`。

| 字段 | 类型 | 说明 |
|------|------|------|
| `model` | string | `SSLR` 或 `SSXGBoost` |
| `n_epochs` | number | SSLR 训练轮数 |
| `batch_size` | number | 批大小 |
| `lr` | number | 学习率 |
| `val_steps` | number | 验证步频 |
| `n_estimators` | number | SSXGBoost 树棵数 |
| `max_depth` | number | 树深度 |
| `K_quantiles` / `k_quantiles` | number | 分位数个数 |
| `reg_coef` | number | 正则系数 |

**返回值**：

- 成功：`{"status":"success","message":"已启动 <model> 训练","training_status":"PSI中"}`（初始状态）
- 失败：如未找到 Deployment、副本为 0、K8s 异常等 → `{"status":"error","message":"..."}`

**注意事项**：

- 训练脚本路径、数据集路径在服务端写死为容器内 `/app/company/...`、`/app/partner/...`。
- 训练完成后逻辑可能尝试将 Deployment `args` 恢复为“仅启动 Ray、不自动训练”的脚本，以便再次发起训练（见源码中 `_deployment_cleaned_after_training` 相关逻辑）。

---

## 3. WebSocket（Socket.IO）事件

### 3.1 客户端连接

**事件**：`connect`（默认命名空间）

**服务端行为**：向该连接 `emit('status_update', { company_status, partner_status, training_status, current_step, total_steps, accuracy, loss })`。

---

### 3.2 服务端推送

| 事件名 | payload 含义 |
|--------|----------------|
| `log_update` | 单条日志 `{timestamp, level, message}` |
| `status_update` | 状态快照（字段同上或仅子集，如仅 `company_status`） |

**注意事项**：与 HTTP 使用同一 Flask 应用；浏览器需加载 Socket.IO 客户端并与 HTTP 同源或正确配置跨域。

---

## 4. 公共基类与工具（`company/common.py`）

### 4.1 `class SSML`

**功能**：秘密共享机器学习模型基类；`SSLR` 与 `SSXGBoost` 均继承自 `SSML`。

**`__init__(devices)`**

| 参数 | 类型 | 说明 |
|------|------|------|
| `devices` | `dict` | 须含 `company`、`partner`（`PYU`）；可选 `spu`（`SPU`） |

**抽象/子类实现的方法**：

| 方法 | 说明 |
|------|------|
| `fit(...)` | 子类实现训练逻辑 |
| `predict(X: FedNdarray, device: PYU) -> PYUObject` | 子类实现推理 |
| `save(paths: dict, ext: str = 'npy')` | 子类实现保存 |
| `load(cls, devices, paths)` | 类方法，子类实现加载 |

**`SSML.score(y_true, y_pred)`（静态方法）**

**功能**：在标签同一 `PYU` 上计算准确率、F1、误漏率（FOR）。

**参数**：`y_true`、`y_pred` 均为 `PYUObject`，且在同一设备。

**返回值**：`dict`，键为：

| 键 | 含义 |
|----|------|
| `accuracy` | 准确率 |
| `f1_metric` | F1 |
| `for_metric` | 误漏率（False omission rate） |

（内部经 `sf.reveal` 为明文。）

---

### 4.2 其他（节选）

| 名称 | 功能 |
|------|------|
| `load_dataset(dataset: str)` | 按数据集名加载训练/测试特征与标签（开发测试用，路径依赖 `Datasets/` 等） |
| `MPCInitializer` | 单例：初始化 SecretFlow 仿真/分布式模式与 `SPU`、PYU 等（见类内 `mode`：`single_sim` / `multi_sim` / `multi_distributed`） |
| `SS_share` | 加法秘密共享拆分 |
| `approx_sigmoid` / `sigmoid` / `softmax` | 激活与损失辅助（供 LR/XGB 内部使用） |

---

## 5. 纵向联邦逻辑回归（`company/LR.py` — `class SSLR`）

**功能**：继承 `SSML`，纵向划分特征下的安全联邦逻辑回归；支持近似 sigmoid（`approx=True`）或精确 sigmoid/softmax（`approx=False`，多分类用 softmax）。

**`__init__(devices, lambda_=0, approx=True)`**

| 参数 | 说明 |
|------|------|
| `devices` | 须含 `spu`、`company`、`partner` |
| `lambda_` | L2 正则系数 |
| `approx` | `True` 时用分段线性近似 sigmoid；`False` 时标签在持有方明文计算梯度（多分类用 softmax） |

**`fit(X, y, X_test=None, y_test=None, batch_size=64, val_steps=1, n_epochs=10, lr=0.1, split_col=None)`**

**功能**：批量梯度下降训练若干 epoch；可选验证集周期性输出 Accuracy / F1 / FOR。

| 参数 | 说明 |
|------|------|
| `X` | `SPUObject`，特征在 SPU 上秘密共享 |
| `y` | `SPUObject` 或 `PYUObject`，标签 |
| `X_test` / `y_test` | 验证集；与 `split_col` 二选一必填逻辑见源码：无验证集则必须提供 `split_col` |
| `batch_size` | 批大小 |
| `val_steps` | 每隔多少 step 验证一次 |
| `n_epochs` | 轮数 |
| `lr` | 初始学习率（实现中按 epoch 衰减） |
| `split_col` | company 与 partner 特征列分界（列索引） |

**返回值**：若启用验证，返回 `accs`（各次验证准确率列表）；训练结束后权重经 `dispatch_weight` 转为各方 `FedNdarray` 明文分块。

**注意事项**：`approx=False` 时要求 `y` 不在 SPU 上（断言见源码）。

**`predict(X, device, threshold=None)`**

**功能**：纵向 `FedNdarray` 特征推理；在 `device` 上输出离散标签（`PYUObject`）。

| 参数 | 说明 |
|------|------|
| `X` | `FedNdarray` |
| `device` | 结果所在 `PYU` |
| `threshold` | 二分类阈值，默认 `0.5` 或模型属性 `pred_threshold` |

**`save(paths: dict[str, str], ext='npy')`**

**功能**：将 `FedNdarray` 权重按 `company`/`partner` 目录分别写入 `weight.npy` 或 `weight.csv` 及 `info.json`（含 `shape`、`lambda_`、`approx` 等）。

**前置条件**：`self.w` 已为 `FedNdarray`（训练结束已转换）。

**`load(cls, devices, paths)`（类方法）**

**功能**：从双方目录读取 `info.json` 与权重文件，重建 `SSLR` 实例与 `FedNdarray` 权重。

**注意事项**：双方 `info.json` 需一致（`assert info1 == info2`）。

---

**说明**：同文件中的 `LR`、`LRSS` 等为本地明文小工具类，非联邦主路径，略。

---

## 6. 纵向联邦 XGBoost（`company/XGBoost.py` — `class SSXGBoost`）

**功能**：基于 SecretFlow 的纵向梯度提升树；内部用 `Tree` / `TreeNode` / `Leaf` 构建单棵树，多棵树集成。

**`__init__(devices, n_estimators=3, lambda_=1e-5, max_depth=3, div=False, mission='Classification')`**

| 参数 | 说明 |
|------|------|
| `devices` | 须含 `spu`、`company`、`partner` |
| `n_estimators` | 树棵数 |
| `lambda_` | 叶子 L2 正则 |
| `max_depth` | 最大深度 |
| `div` | 是否用除法计算叶子权重与增益 |
| `mission` | `'Classification'` 或 `'Regression'` |

**`fit(X, y, buckets, FedQuantiles, X_test=None, y_test=None)`**

**功能**：顺序训练多棵树；标签在 `PYU`（`y` 持有者）。

| 参数 | 说明 |
|------|------|
| `X` | `SPUObject`，训练特征秘密共享 |
| `y` | `PYUObject`，明文标签 |
| `buckets` | `np.ndarray`，桶划分（公开） |
| `FedQuantiles` | `FedNdarray`，各方分位点 |
| `X_test` / `y_test` | 可选，验证集 |

**返回值**：`(train_accs, test_accs)`（列表，可能为空列表）；终端打印各 Step 的 Accuracy / F1 / FOR 及最终汇总。

**`predict(X, device) -> PYUObject`**

**功能**：`X` 为 `FedNdarray` 或 `SPUObject`（训练阶段/评估阶段不同）；前向求和后经激活与 `to_int_labels` 得到类别。

**`save(paths, ext='npy')`**

**功能**：各方目录写入 `weight.npy`、`quantiles.npy`（或 CSV 变体）、`info.json`、`tree.pkl`（`dill` 序列化树结构）；叶子权重经秘密共享拆分后分别落盘。

**`load(cls, devices, paths) -> SSXGBoost`**

**功能**：从双方目录恢复模型、分位点与树结构；**必须**提供含 `spu` 的 `devices`（与训练一致）。

---

## 7. 推理编排（`company/infer.py` — `class InferEngine`）

**功能**：在仅初始化 `PYU`（及可选 `SPU`）的前提下，加载 `SSLR` 或 `SSXGBoost`、读 CSV、求交、构造纵向 `FedNdarray`、调用 `predict` 并封装为带 `id` 的 `DataFrame`。

**`__init__(devices, model)`**

| 参数 | 说明 |
|------|------|
| `devices` | 须含 `company`、`partner`、`coordinator`（均为 `PYU`）；可选 `spu`（`SPU`） |
| `model` | `'SSLR'` 或 `'SSXGBoost'` |

**`load_model(paths)`**

**功能**：调用 `SSLR.load` 或 `SSXGBoost.load`，传入 `{'company': company, 'partner': partner}` 与各方模型目录路径。

**返回值**：`self.model`。

**`load_data_from_path(paths)`**

**返回**：`company_keys, company_data, partner_keys, partner_data`（前两列 id 为 `iloc[:,0]`）。

**`compute_intersection(...)`**

**功能**：coordinator 求 id 交集；过滤排序后去 id，若存在 `Revenue` 列则删除；`load` 为纵向 `FedNdarray` → `self.X`。

**`infer(device)`**

**功能**：`self.model.predict(self.X, device=device)`，再包装为 `pd.DataFrame({'id','prediction'})`。

**`InferEngine.score(y_true, y_pred)`（静态方法）**

**功能**：转发 `SSML.score`；参数需满足 `SSML.score` 对 `PYUObject` 的要求（与 `infer.py` 中 `example_run` 用法一致时需在同一设备上）。

---

## 8. 相关配置与常量

- **Web UI 配置路径**：`web_ui/config.yaml`（IP、端口、训练默认值等）。
- **K8s 标签**：Company `app.kubernetes.io/component=company`，Partner `app.kubernetes.io/component=partner`。
- **network_utils 中的 HTTP 约定**：`POST http://{ip}:{port}/api/{command}` 为工具层预留调用形式；当前 Web UI 路由以本文第 2 节为准，若扩展需与 `NetworkManager._send_http_command` 对齐。

---

## 9. 部署与测试

部署与联调流程可参考项目内 **`操作手册.md`**、**`DEPLOYMENT.md`**、**`QUICKSTART.md`** 及 Helm Chart `helm-chart/mobile-mpc-project/`。

**快速自检示例**（Web UI 已监听时）：

```bash
curl -s "http://127.0.0.1:8080/api/status"
curl -s "http://127.0.0.1:8080/api/training_metrics"
curl -s "http://127.0.0.1:8080/api/logs/company"
```

（端口以实际运行或 Service 为准。）

---

## 10. 代码对照说明

| 文档章节 | 主要源码文件 |
|----------|----------------|
| Web UI HTTP / SocketIO | `web_ui/app.py` |
| 基类与评估 | `company/common.py`（`SSML` 等） |
| 逻辑回归联邦模型 | `company/LR.py`（`SSLR`） |
| XGBoost 联邦模型 | `company/XGBoost.py`（`SSXGBoost`，内部 `Tree` 等） |
| 推理编排 | `company/infer.py`（`InferEngine`） |
| 训练入口 | `company/truerun.py` |
| Service 端口 | `helm-chart/mobile-mpc-project/values.yaml`（`webui.service`） |

**说明**：脚本 `test_model_params_feature.sh` 若引用 `GET /api/models/list` 等路由，请以当前 `web_ui/app.py` 实际路由为准并同步更新本文第 2 节。

---

*文档版本：与仓库 `mobile_project3_new` 中 `web_ui/app.py`、`company/common.py`、`LR.py`、`XGBoost.py`、`infer.py` 对照整理。*
