# 项目路径
PROJ_PATH="/home/dxn/mobile_project2/AnonymVFL"
cd ${PROJ_PATH}
python3 ${PROJ_PATH}/AnonymVFL/run.py \
    --mode="multi_sim" \
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
    --batch_size=1000 \
    --val_steps=1 \
    --lr=0.1 \
    --path_to_company_model_save_dir="${PROJ_PATH}/models/lr_company" \
    --path_to_partner_model_save_dir="${PROJ_PATH}/models/lr_partner" \
    2> >(grep -v -E "(openssl_factory|Yacl has been configured|entropy.*source)" >&2)
