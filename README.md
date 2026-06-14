# AnonymVFL — 匿名化垂直联邦学习平台

<p align="center">
  <strong>基于 SecretFlow 的隐私保护多方安全计算（MPC）垂直联邦学习系统</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue.svg" alt="Python 3.10">
  <img src="https://img.shields.io/badge/SecretFlow-1.12.0b0-green.svg" alt="SecretFlow 1.12.0b0">
  <img src="https://img.shields.io/badge/Kubernetes-Ready-326ce5.svg" alt="Kubernetes Ready">
</p>

---

## 📖 目录

- [AnonymVFL — 匿名化垂直联邦学习平台](#anonymvfl--匿名化垂直联邦学习平台)
  - [📖 目录](#-目录)
  - [项目简介](#项目简介)
    - [什么是垂直联邦学习？](#什么是垂直联邦学习)
    - [核心技术栈](#核心技术栈)
  - [核心特性](#核心特性)
  - [系统架构](#系统架构)
  - [支持的算法](#支持的算法)
    - [1. SSLR（Secret-Shared Logistic Regression）](#1-sslrsecret-shared-logistic-regression)
    - [2. SSXGBoost（Secret-Shared XGBoost）](#2-ssxgboostsecret-shared-xgboost)
  - [核心模块代码结构](#核心模块代码结构)
  - [项目结构](#项目结构)
  - [快速开始](#快速开始)
    - [环境要求](#环境要求)
    - [本地单机模拟](#本地单机模拟)
    - [多机分布式部署](#多机分布式部署)
    - [Kubernetes 部署](#kubernetes-部署)
  - [推理（Inference）](#推理inference)
    - [本地单机模拟](#本地单机模拟-1)
    - [多机分布式部署](#多机分布式部署-1)
    - [Kubernetes 部署](#kubernetes-部署-1)
    - [推理流程](#推理流程)
  - [API 接口文档](#api-接口文档)
  - [主要参考资料](#主要参考资料)
  - [实验](#实验)

---

## 项目简介

AnonymVFL 是一个**生产级垂直联邦学习（Vertical Federated Learning, VFL）平台**，基于蚂蚁集团开源的 [SecretFlow](https://github.com/secretflow/secretflow) 多方安全计算框架构建。它允许多个参与方在不共享原始数据的前提下，联合训练机器学习模型。

### 什么是垂直联邦学习？

在垂直联邦学习场景中，不同机构持有同一批样本的**不同特征列**（即数据是垂直分割的）。例如：

- **甲方（Company）**：持有用户的基本信息 + 标签（如是否违约）
- **乙方（Partner）**：持有同一批用户的交易行为特征

通过本平台，双方可以在**不暴露各自原始数据**的情况下，共同训练一个完整的模型。

### 核心技术栈

| 组件 | 用途 |
|------|------|
| [SecretFlow](https://secretflow.readthedocs.io/) | 隐私保护多方安全计算框架 |
| SPU (Secure Processing Unit) | 安全多方计算执行单元，协议3/域3 |
| HEU (Homomorphic Encryption Unit) | 同态加密单元（PSI阶段使用） |
| [Ray](https://docs.ray.io/) | 分布式计算编排 |
| [JAX](https://jax.readthedocs.io/) | 高性能数值计算与自动微分 |
| Ristretto255 (rbcl) | 椭圆曲线密码学（PSI 阶段） |
| Flask + SocketIO | Web 管理后台 |
| Kubernetes + Helm | 容器化编排与多集群部署 |

---

## 核心特性

✅ **隐私优先** — 原始数据不出域，基于密码学协议保证中间计算结果不泄露

✅ **多种算法支持** — 内置 SSLR（秘密共享逻辑回归）和 SSXGBoost（秘密共享 XGBoost）

✅ **私有集合求交（PSI）** — 基于 Ristretto255 椭圆曲线，安全找出双方共有样本

✅ **Web 可视化管理** — 甲方可发起训练/推理，双方均可实时监控训练指标

✅ **灵活部署** — 支持本地单机模拟、多机分布式、Kubernetes 多集群三种模式

✅ **模型持久化** — 权重按参与方分区存储，支持模型版本管理

✅ **批量推理** — 支持加载已训练模型对新数据进行安全推理

---

## 系统架构

```
┌──────────────────────────────────────────────────────────┐
│                    Web UI (Flask)                        │
│          http://<company-ip>:5000 (或 :8080)             │
└──────────────┬───────────────────────┬───────────────────┘
               │                       │
┌──────────────▼──────────┐  ┌─────────▼──────────────────┐
│     Company Node (甲方)   │  │    Partner Node (乙方)      │
│                          │  │                            │
│  ┌────────────────────┐  │  │  ┌──────────────────────┐  │
│  │  Ray Head Node     │◄─┼──┼──│  Ray Worker Node     │  │
│  │  (port 6379/20001) │  │  │  └──────────────────────┘  │
│  └────────────────────┘  │  │                            │
│  ┌────────────────────┐  │  │  ┌──────────────────────┐  │
│  │  SPU (port 11001)  │◄─┼──┼──│  SPU (port 11002)    │  │
│  └────────────────────┘  │  │  └──────────────────────┘  │
│  ┌────────────────────┐  │  │                            │
│  │  Coordinator SPU   │  │  │  持有特征数据（无标签）      │
│  │  (port 11003)      │  │  │  只读 WebUI               │
│  └────────────────────┘  │  │                            │
│                          │  │                            │
│  持有特征数据 + 标签       │  │                            │
│  完全控制 WebUI           │  │                            │
└──────────────────────────┘  └────────────────────────────┘
```

## 支持的算法

### 1. SSLR（Secret-Shared Logistic Regression）

基于秘密共享的逻辑回归，支持二分类和多分类。

- `--share_y True` 时：标签秘密共享到 SPU，使用近似 Sigmoid 函数（MPC 友好）
- `--share_y False`（默认）时：标签保留在主动方 PYU，使用精确 Sigmoid/Softmax

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--n_epochs` | 训练轮数 | 10 |
| `--lr` | 学习率 | 0.1 |
| `--batch_size` | 批次大小 | 1024 |
| `--reg_coef` | L2 正则化系数 | 1e-5 |
| `--val_steps` | 验证频率（每 N 个 epoch） | 1 |
| `--share_y` | 是否秘密共享标签 y | False |

### 2. SSXGBoost（Secret-Shared XGBoost）

基于秘密共享的梯度提升树。

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--n_estimators` | 树的数量 | 2 |
| `--max_depth` | 树的最大深度 | 2 |
| `--K_quantiles` | 特征分桶数 | 10 |
| `--reg_coef` | 正则化系数 | 1e-5 |
| `--gamma` | 信息增益阈值 | 0.0 |

---

## 核心模块代码结构

三个核心模块均位于 `company/` 目录下，分别实现私有集合求交、秘密共享逻辑回归和秘密共享 XGBoost。

### PSI.py — 私有集合求交

- **`PSIWorker`** — 基类，负责读取己方数据（键值 + 私有特征 + 可选的公开特征），并对键值进行哈希-乘方处理、对特征随机排列
- **`PSICompany`** — 甲方求交逻辑：`exchange()` 将哈希键值和加密特征发送给乙方；`compute_intersection()` 比较双方二次乘方哈希值求交集，并计算己方特征分片
- **`PSIPartner`** — 乙方求交逻辑：`exchange()` 对甲方密文加减掩码后重排；`output_shares()` 输出乙方特征分片
- **`private_set_intersection()`** — 顶层接口，串联双方交互流程，返回双方的特征分片与可选的分桶标签

### LR.py — 秘密共享逻辑回归

- **`SSLR`**（继承 `SSML`） — 支持二分类/多分类，提供 `approx` 参数控制是否使用近似 Sigmoid
  - `fit()` — 训练入口：数据分 batch 后迭代执行前向 + 反向传播，训练完成后 `dispatch_weight()` 将权重分发到各方
  - `_forward()` / `_backward()` — 前向计算预测值、反向计算梯度并更新权重
  - `predict()` — 推理：各方本地计算特征与权重内积后聚合，经激活函数输出类别
  - `save()` / `load()` — 权重按方分区存储（npy/csv）及加载
- **`SSLR_test()`** — 独立测试函数，用于与 sklearn LogisticRegression 对比

### XGBoost.py — 秘密共享 XGBoost

- **`TreeNode` / `Leaf`** — 树节点数据结构，内部节点存分裂阈值（分位点索引），叶子节点存权重索引
- **`Tree`**（继承 `SSML`） — 单棵决策树的构建与推理
  - `fit()` — 计算一/二阶梯度后调用 `_build_tree()` 递归建树
  - `_build_tree()` — 递归分裂：`_aggregate_bucket()` 聚合桶内梯度，`_split()` 用淘汰赛两两比较法寻找最优分裂点，`_leaf()` 计算叶子权重
  - `forward()` — 对联邦特征逐节点判断归属，输出叶子权重
- **`SSXGBoost`**（继承 `SSML`） — 多棵树的集成模型
  - `fit()` — 迭代训练多棵 `Tree`，累加预测值，记录训练/验证指标
  - `predict()` — 累加每棵树输出后经激活函数输出类别
  - `save()` / `load()` — 保存/加载树结构与分位点，权重存于标签持有方
- **`quantize_buckets()` / `recover_buckets()`** — 分桶工具：对原始特征等频分桶生成桶列表，PSI 后从标签矩阵恢复桶列表
- **`SSXGBoost_test()`** — 独立测试函数

---

## 项目结构

```
AnonymVFL/
│
├── 📄 文档
│   ├── DEPLOYMENT.md                   # K8s 部署指南
│   ├── QUICKSTART.md                   # 5 分钟快速上手
│   ├── TEST_FLOW.md                    # 双集群测试流程
│   ├── LOCAL_INFER_REPRO.md            # 本地推理调试指南
│   ├── 操作手册.md / 操作手册.pdf        # 原始操作手册
│   └── 联邦学习-WebUI-接口文档.md        # WebUI API 文档
│
├── 🏢 company/                         # 甲方节点（Server）
│   ├── truerun.py                      # 训练入口脚本
│   ├── infer_run.py                    # 推理入口脚本
│   ├── common.py                       # 公共工具类（MPCInitializer, SSML 基类等）
│   ├── LR.py                           # SSLR 逻辑回归实现
│   ├── XGBoost.py                      # SSXGBoost 实现
│   ├── PSI.py                          # 私有集合求交实现
│   ├── infer.py                        # 推理引擎
│   ├── startA.sh                       # 一键启动（Ray Head + 训练）
│   ├── start_ray_only.sh               # 仅启动 Ray Head
│   ├── start_ray_with_log.sh           # 启动 Ray Head（带日志）
│   ├── start_training.sh               # 训练启动脚本（需 Ray 已启动）
│   ├── start_training_with_log.sh      # 训练启动脚本（带日志）
│   ├── start_infer.sh                  # 推理启动脚本（需 Ray 已启动）
│   ├── single_run.sh / single_infer.sh # 单机模拟快捷脚本
│   ├── k8s_start_training.sh           # K8s 训练触发脚本
│   ├── k8s_start_infer.sh              # K8s 推理触发脚本
│   ├── clean.sh                        # 清理脚本
│   ├── host_train.csv / host_test.csv  # 示例训练/测试数据
│   ├── Datasets/PSI/                   # PSI 测试数据
│   └── models/                         # 训练产出模型目录（lr_company/, xgb_company/）
│
├── 🤝 partner/                         # 乙方节点（Client）
│   ├── joinB.py                        # 乙方入口（加入 Ray 集群）
│   ├── joinB.sh / join_with_log.sh     # 启动脚本
│   ├── clean.sh                        # 清理脚本
│   └── guest_train.csv / guest_test.csv # 示例训练/测试数据
│
├── 🌐 web_ui/                          # Web 管理界面
│   ├── app.py                          # Flask 主应用
│   ├── config.yaml                     # 配置文件（IP、端口、训练参数）
│   ├── network_utils.py                # K8s API 工具
│   ├── test_network.py / test_subprocess.py # 测试脚本
│   ├── requirements.txt                # Web UI 依赖
│   ├── run.sh / start_web_ui.sh        # 启动脚本
│   └── templates/index.html            # 前端页面
│
├── 🐳 部署相关
│   ├── Dockerfile / Dockerfile.company / Dockerfile.partner
│   ├── requirements.txt                # Python 依赖
│   ├── build-multi-cluster-images.sh   # 构建镜像
│   ├── deploy-multi-cluster.sh         # 多集群部署
│   ├── deploy-steps.sh                 # 分步部署
│   ├── rebuild_and_deploy.sh           # 重建并部署
│   ├── validate-chart.sh               # Helm Chart 校验
│   └── helm-chart/mobile-mpc-project/  # Helm Chart
│       ├── Chart.yaml
│       ├── values.yaml                 # 默认配置
│       ├── values-cluster-a.yaml       # 甲方集群配置
│       ├── values-cluster-b.yaml       # 乙方集群配置
│       └── templates/                  # K8s 资源模板
│
└── 📊 测试
    ├── test_model_params_feature.sh    # 模型参数测试
    └── data_full.csv                   # 完整测试数据集
```

---

## 快速开始

### 环境要求

- **Python**: 3.10
- **操作系统**: Linux（推荐 Ubuntu 20.04+）
- **内存**: ≥ 16 GB（单机模拟模式需要较大内存用于 HEU 初始化）
- **网络**: 多机模式下各节点需网络互通

### 本地单机模拟

适用于开发调试，在同一进程中模拟双方。

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 运行训练（SSLR）
cd company
python truerun.py \
  --mode single_sim \
  --run_psi True \
  --model SSLR \
  --n_epochs 10 \
  --lr 0.1 \
  --batch_size 1024 \
  --path_to_company_train_dataset <company_train.csv> \
  --path_to_partner_train_dataset <partner_train.csv> \
  --path_to_company_val_dataset <company_val.csv> \
  --path_to_partner_val_dataset <partner_val.csv>

# 3. 或使用快捷脚本
bash single_run.sh
```

### 多机分布式部署

适用于两台物理机或虚拟机。通过 `start_ray_only.sh`、`start_training.sh` 和 `joinB.sh` 脚本分别启动双方节点。

**前置步骤：** 修改三个脚本中的 IP 地址为实际机器 IP。

- `company/start_ray_only.sh`：修改 `A_IP`
- `company/start_training.sh`：修改 `A_IP` 和 `B_IP`
- `partner/joinB.sh`：修改 `A_IP`

**机器 A（Company）— 先启动 Ray Head：**

```bash
cd company
bash start_ray_only.sh
```

该脚本启动 Ray Head 节点（监听 `20001` 端口），注册 `company` 和 `coordinator` 资源，并保持运行。

**机器 B（Partner）— 加入 Ray 集群：**

```bash
cd partner
bash joinB.sh
```

该脚本以 Worker 身份加入 Ray 集群，注册 `partner` 资源，并保持运行等待任务分配。

**机器 A（Company）— 启动训练：**

```bash
cd company
bash start_training.sh
```

此外还有一种两步启动的方式，请参阅[操作手册.md](操作手册.md)。

### Kubernetes 部署

适用于生产环境的多集群部署。详细部署说明请参阅 [DEPLOYMENT.md](DEPLOYMENT.md) 和 [QUICKSTART.md](QUICKSTART.md)。

---

## 推理（Inference）

使用已训练的模型对新数据进行安全推理，支持三种部署模式。

### 本地单机模拟

```bash
cd company
bash single_infer.sh
```

或手动指定参数：

```bash
cd company
python infer_run.py \
  --mode single_sim \
  --model SSLR \
  --company_model_path <company_model_dir> \
  --partner_model_path <partner_model_dir> \
  --company_data_path <company_infer_data.csv> \
  --partner_data_path <partner_infer_data.csv>
```

### 多机分布式部署

**前置步骤：** 修改 `company/start_infer.sh` 中的 `A_IP` 和 `B_IP` 为实际机器 IP。

**机器 A（Company）— 先启动 Ray Head：**

```bash
cd company
bash start_ray_only.sh
```

**机器 B（Partner）— 加入 Ray 集群：**

```bash
cd partner
bash joinB.sh
```

**机器 A（Company）— 执行推理：**

```bash
cd company
bash start_infer.sh
```

### Kubernetes 部署

在 Company Pod 内执行：

```bash
cd company
bash k8s_start_infer.sh
```

该脚本会自动通过环境变量（`POD_IP`、`PARTNER_SPU_ADDR` 等）获取 Pod IP 和 Partner 地址，无需手动配置。

### 推理流程

1. 加载 Company 和 Partner 各自的模型权重分区
2. 对推理数据执行普通求交
3. 在 SPU 中重构完整模型并执行安全前向传播
4. Company 获取并返回预测结果

---

## API 接口文档

WebUI 提供的 REST API 接口详见 [联邦学习-WebUI-接口文档.md](联邦学习-WebUI-接口文档.md)。

## 主要参考资料

- [SecretFlow 官方文档](https://secretflow.readthedocs.io/)
- [SecureML: A System for Scalable Privacy-Preserving Machine Learning](https://www.ieee-security.org/TC/SP2017/papers/466.pdf)
- [An Efficient Learning Framework for Federated XGBoostUsing Secret Sharing and Distributed Optimization](https://dl.acm.org/doi/epdf/10.1145/3523061)

---
## 实验
实验代码及实验结果请移步`test`分支。