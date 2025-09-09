#!/bin/bash

# Coordinator节点
# 项目路径
PROJ_PATH="/home/dxn/mobile_project2/AnonymVFL"
cd ${PROJ_PATH}

# 连接到Ray集群
echo "Connecting to Ray cluster..."
ray start --address=210.28.133.104:20001 --num-cpus=8 --resources='{"coordinator": 10}' --node-ip-address=210.28.133.106

echo "Connected to Ray cluster as Coordinator node"
echo "Starting coordination for federated learning..."
sleep 5

# 运行Coordinator节点的联邦学习任务
python3 ${PROJ_PATH}/AnonymVFL/run.py \
    --mode="multi_distributed" \
    --party="coordinator" \
    --ray_head_addr="210.28.133.104:20001" \
    --company_spu_addr="210.28.133.104:11001" \
    --partner_spu_addr="210.28.133.105:11002" \
    --coordinator_spu_addr="210.28.133.106:11003" \
    --run_psi=True \
    --path_to_company_train_dataset="${PROJ_PATH}/Datasets/data/data/host_train.csv" \
    --path_to_partner_train_dataset="${PROJ_PATH}/Datasets/data/data/guest_train.csv" \
    --path_to_company_share="${PROJ_PATH}/company_share.csv" \
    --path_to_partner_share="${PROJ_PATH}/partner_share.csv" \
    --share_y=False \
    --path_to_company_val_dataset="${PROJ_PATH}/Datasets/data/data/host_test.csv" \
    --path_to_partner_val_dataset="${PROJ_PATH}/Datasets/data/data/guest_test.csv" \
    --model="SSLR" \
    --n_epochs=10 \
    --batch_size=1024 \
    --val_steps=1 \
    --lr=0.1 \
    --path_to_company_model_save_dir="${PROJ_PATH}/models/lr_company" \
    --path_to_partner_model_save_dir="${PROJ_PATH}/models/lr_partner"

echo "Coordinator training completed. Disconnecting from Ray cluster..."
ray stop
