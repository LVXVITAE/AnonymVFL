# 本地复现文档：推理维度不一致问题（不依赖 K8s）

## 1. 目标

在本地快速复现并定位推理报错（例如 `size 8 is different from 9`），先排除 K8s 网络/部署干扰，确认是否为数据或特征维度问题。

---

## 2. 前置条件

- 项目目录：`/home/dxn/mobile_project_final/mobile_project3_new`
- Python 环境已安装项目依赖（含 `secretflow`）
- 使用仓库自带数据：
  - `company/host_train.csv`
  - `company/host_test.csv`
  - `partner/guest_train.csv`
  - `partner/guest_test.csv`

---

## 3. 最小复现流程（single_sim）

先进入目录：

```bash
cd /home/dxn/mobile_project_final/mobile_project3_new/company
```

### 3.1 本地训练（single_sim）

```bash
python truerun.py \
  --mode single_sim \
  --run_psi True \
  --path_to_company_train_dataset ./host_train.csv \
  --path_to_partner_train_dataset ../partner/guest_train.csv \
  --path_to_company_val_dataset ./host_test.csv \
  --path_to_partner_val_dataset ../partner/guest_test.csv \
  --path_to_company_share ./company_share.csv \
  --path_to_partner_share ../partner/partner_share.csv \
  --path_to_company_model_save_dir ./models/lr_company \
  --path_to_partner_model_save_dir ../partner/models/lr_partner \
  --model SSLR \
  --n_epochs 10 \
  --batch_size 1000 \
  --val_steps 1 \
  --lr 0.1
```

### 3.2 本地推理（single_sim）

```bash
python infer_run.py \
  --mode single_sim \
  --model SSLR \
  --company_model_path ./models/lr_company \
  --partner_model_path ../partner/models/lr_partner \
  --company_data_path ./host_test.csv \
  --partner_data_path ../partner/guest_test.csv
```

预期：
- 成功时，最后一行是 JSON：`{"status":"success",...}`
- 失败时，通常会直接打印栈信息（例如矩阵维度不匹配）

---

## 4. 维度排查命令（关键）

### 4.1 检查 train/test 列是否一致（Company + Partner）

```bash
python - << 'PY'
import pandas as pd

def check(name, train_path, test_path):
    tr = pd.read_csv(train_path)
    te = pd.read_csv(test_path)
    print(f"\n[{name}]")
    print("train shape:", tr.shape, "test shape:", te.shape)
    print("only in train:", sorted(set(tr.columns)-set(te.columns)))
    print("only in test :", sorted(set(te.columns)-set(tr.columns)))
    print("same order:", list(tr.columns) == list(te.columns))

check("company", "./host_train.csv", "./host_test.csv")
check("partner", "../partner/guest_train.csv", "../partner/guest_test.csv")
PY
```

### 4.2 检查模型权重维度

```bash
python - << 'PY'
import numpy as np, json

w = np.loadtxt("./models/lr_company/weight.csv", delimiter=",")
print("company weight shape:", w.shape)

with open("./models/lr_company/info.json","r") as f:
    info = json.load(f)
print("company info.json:", info)
PY
```

---

## 5. 结果判定

- 若本地也报 `size 8 is different from 9`：
  - 说明核心问题是训练/推理特征维度不一致（与 K8s 无关）
- 若本地成功但 K8s 失败：
  - 说明还叠加了 K8s 地址注入或运行参数问题（如 `ray_head_addr`、`partner_spu_addr`）

---

## 6. 建议的排错顺序

1. 先用 `single_sim` 复现并修复维度问题  
2. 本地推理稳定后，再回到 K8s 做双集群联调  
3. K8s 中重点核对推理命令里的四个地址：  
   - `ray_head_addr`
   - `company_spu_addr`
   - `partner_spu_addr`
   - `coordinator_spu_addr`

