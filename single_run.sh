# 项目路径
PROJ_PATH="/home/lvx_vitae/AnonymVFL"
cd ${PROJ_PATH}
# python ${PROJ_PATH}/AnonymVFL/run.py \
#     --mode="single_sim" \
#     --run_psi=True \
#     --path_to_company_train_dataset="${PROJ_PATH}/Datasets/data/data/breast_hetero_host.csv" \
#     --path_to_partner_train_dataset="${PROJ_PATH}/Datasets/data/data/breast_hetero_guest.csv" \
#     --path_to_company_share="${PROJ_PATH}/company_share.csv" \
#     --path_to_partner_share="${PROJ_PATH}/partner_share.csv" \
#     --share_y=False \
#     --path_to_company_val_dataset="${PROJ_PATH}/Datasets/data/data/breast_hetero_host_test.csv" \
#     --path_to_partner_val_dataset="${PROJ_PATH}/Datasets/data/data/breast_hetero_guest_test.csv" \
#     --model="SSLR" \
#     --n_epochs=10 \
#     --batch_size=1024 \
#     --val_steps=1 \
#     --lr=0.1 \
#     --path_to_company_model_save_dir="${PROJ_PATH}/models/lr_company" \
#     --path_to_partner_model_save_dir="${PROJ_PATH}/models/lr_partner"

python ${PROJ_PATH}/AnonymVFL/run.py \
    --mode="single_sim" \
    --run_psi=True \
    --path_to_company_train_dataset="${PROJ_PATH}/Datasets/data/data/breast_hetero_host.csv" \
    --path_to_partner_train_dataset="${PROJ_PATH}/Datasets/data/data/breast_hetero_guest.csv" \
    --path_to_company_share="${PROJ_PATH}/company_share.csv" \
    --path_to_partner_share="${PROJ_PATH}/partner_share.csv" \
    --share_y=False \
    --path_to_company_val_dataset="${PROJ_PATH}/Datasets/data/data/breast_hetero_host_test.csv" \
    --path_to_partner_val_dataset="${PROJ_PATH}/Datasets/data/data/breast_hetero_guest_test.csv" \
    --model="SSXGBoost" \
    --n_estimators=2 \
    --max_depth=2 \
    --K_quantiles=20