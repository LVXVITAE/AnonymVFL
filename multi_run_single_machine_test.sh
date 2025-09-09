#!/bin/bash

# 单机三节点测试脚本
# 在同一台机器上启动三个进程模拟三个节点

PROJ_PATH="/home/dxn/mobile_project2/AnonymVFL"
cd ${PROJ_PATH}

# 清理之前的Ray集群
echo "Cleaning previous Ray cluster..."
ray stop --force 2>/dev/null || true
sleep 2

# 启动Ray Head节点
echo "Starting Ray Head node..."
ray start --head --port=20001 --num-cpus=24 --resources='{"company": 10, "partner": 10, "coordinator": 10}' --node-ip-address=210.28.133.104 &
sleep 5

# 启动Company节点进程
echo "Starting Company node process..."
python3 ${PROJ_PATH}/AnonymVFL/run.py \
    --mode="multi_distributed" \
    --party="company" \
    --ray_head_addr="210.28.133.104:20001" \
    --company_spu_addr="210.28.133.104:11001" \
    --partner_spu_addr="210.28.133.104:11002" \
    --coordinator_spu_addr="210.28.133.104:11003" \
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
    --path_to_partner_model_save_dir="${PROJ_PATH}/models/lr_partner" &

COMPANY_PID=$!

# 启动Partner节点进程
echo "Starting Partner node process..."
python3 ${PROJ_PATH}/AnonymVFL/run.py \
    --mode="multi_distributed" \
    --party="partner" \
    --ray_head_addr="210.28.133.104:20001" \
    --company_spu_addr="210.28.133.104:11001" \
    --partner_spu_addr="210.28.133.104:11002" \
    --coordinator_spu_addr="210.28.133.104:11003" \
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
    --path_to_partner_model_save_dir="${PROJ_PATH}/models/lr_partner" &

PARTNER_PID=$!

# 启动Coordinator节点进程
echo "Starting Coordinator node process..."
python3 ${PROJ_PATH}/AnonymVFL/run.py \
    --mode="multi_distributed" \
    --party="coordinator" \
    --ray_head_addr="210.28.133.104:20001" \
    --company_spu_addr="210.28.133.104:11001" \
    --partner_spu_addr="210.28.133.104:11002" \
    --coordinator_spu_addr="210.28.133.104:11003" \
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
    --path_to_partner_model_save_dir="${PROJ_PATH}/models/lr_partner" &

COORDINATOR_PID=$!

echo "All nodes started. PIDs: Company=$COMPANY_PID, Partner=$PARTNER_PID, Coordinator=$COORDINATOR_PID"
echo "Waiting for training to complete..."

# 等待所有进程完成
wait $COMPANY_PID
wait $PARTNER_PID
wait $COORDINATOR_PID

echo "Training completed on all nodes!"
echo "Stopping Ray cluster..."
ray stop

echo "All done!"
