#!/usr/bin/env bash
set -euo pipefail

# === A = company + coordinator (head 节点) ===
A_IP="210.28.133.104"
B_IP="210.28.133.104"   # 同机模拟时也用 127.0.0.1 或本机 IP
PORT_RAY="20001"
PORT_COMPANY_SPU="11001"
PORT_PARTNER_SPU="11002"
PORT_COORD_SPU="11003"

cd "$(dirname "$0")"
COM_PATH="."
PAR_PATH="../partner"

# 1) 启动 Ray head（只打 company/coordinator 资源）
# ray stop || true
ray start --head --node-ip-address "${A_IP}" --port "${PORT_RAY}" \
  --num-cpus 8 \
  --resources='{"company": 10, "coordinator": 10}' \
  --object-store-memory=2000000000

# 2) 启动任务
python3 truerun.py \
  --mode="multi_distributed" \
  --ray_head_addr="${A_IP}:${PORT_RAY}" \
  --company_spu_addr="${A_IP}:${PORT_COMPANY_SPU}" \
  --partner_spu_addr="${B_IP}:${PORT_PARTNER_SPU}" \
  --coordinator_spu_addr="${A_IP}:${PORT_COORD_SPU}" \
  --run_psi=True \
  \
  --path_to_company_train_dataset="${COM_PATH}/host_train.csv" \
  --path_to_company_val_dataset="${COM_PATH}/host_test.csv" \
  --path_to_company_share="${COM_PATH}/company_share.csv" \
  --path_to_company_model_save_dir="${COM_PATH}/models/lr_company" \
  --share_y=False \
  \
  --path_to_partner_train_dataset="${PAR_PATH}/guest_train.csv" \
  --path_to_partner_val_dataset="${PAR_PATH}/guest_test.csv" \
  --path_to_partner_share="${PAR_PATH}/partner_share.csv" \
  --path_to_partner_model_save_dir="${PAR_PATH}/models/lr_partner" \
  \
  --model="SSLR" \
  --n_epochs=10 \
  --batch_size=1000 \
  --val_steps=1 \
  --lr=0.1 \
  2> >(grep -v -E "(openssl_factory|Yacl has been configured|entropy.*source)" >&2)
