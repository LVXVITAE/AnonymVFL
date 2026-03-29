# 项目路径
PROJ_PATH="$(dirname "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)")"
COM_PATH="${PROJ_PATH}/company"
PAR_PATH="${PROJ_PATH}/partner"
cd ${PROJ_PATH}

python ${COM_PATH}/truerun.py \
    --path_to_company_train_dataset="${COM_PATH}/host_train.csv" \
    --path_to_partner_train_dataset="${PAR_PATH}/guest_train.csv" \
    --path_to_company_val_dataset="${COM_PATH}/host_test.csv" \
    --path_to_partner_val_dataset="${PAR_PATH}/guest_test.csv" \
    --share_y=True \
    --model="SSLR" \
    --n_estimators=2 \
    --max_depth=2