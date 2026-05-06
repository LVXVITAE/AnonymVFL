#!/usr/bin/env bash
set -euo pipefail

# === A = Company 所在机器（Ray head + 训练入口）===
RAY_HEAD_HOST="${RAY_HEAD_HOST:-210.28.133.104}"
RAY_PORT="${RAY_PORT:-20001}"
COMPANY_SPU_ADDR="${COMPANY_SPU_ADDR:-210.28.133.104:11001}"
PARTNER_SPU_ADDR="${PARTNER_SPU_ADDR:-210.28.133.105:11002}"
COORDINATOR_SPU_ADDR="${COORDINATOR_SPU_ADDR:-210.28.133.106:11003}"

cd "$(dirname "$0")"
COM_PATH="."
PAR_PATH="../partner"

# 1) 启动 Ray head（只打 company 资源）
# ray stop || true
ray start --head --node-ip-address "${RAY_HEAD_HOST}" --port "${RAY_PORT}" \
  --num-cpus 8 \
  --resources='{"company": 10}' \
  --object-store-memory=2000000000

echo "Ray Head: ${RAY_HEAD_HOST}:${RAY_PORT}"
echo "Company SPU: ${COMPANY_SPU_ADDR}"
echo "Partner SPU: ${PARTNER_SPU_ADDR}"
echo "Coordinator SPU: ${COORDINATOR_SPU_ADDR}"

# 2) 启动任务
python3 truerun.py \
  --mode="multi_distributed" \
  --ray_head_addr="${RAY_HEAD_HOST}:${RAY_PORT}" \
  --company_spu_addr="${COMPANY_SPU_ADDR}" \
  --partner_spu_addr="${PARTNER_SPU_ADDR}" \
  --coordinator_spu_addr="${COORDINATOR_SPU_ADDR}" \
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
