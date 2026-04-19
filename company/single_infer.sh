# 项目路径
PROJ_PATH="$(dirname "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)")"
COM_PATH="${PROJ_PATH}/company"
PAR_PATH="${PROJ_PATH}/partner"
cd ${PROJ_PATH}

python ${COM_PATH}/infer_run.py \
    --mode="single_sim" \
    --model="SSLR" \
    --company_model_path="${COM_PATH}/models/lr_company" \
    --partner_model_path="${PAR_PATH}/models/lr_partner" \
    --company_data_path="${COM_PATH}/host_test.csv" \
    --partner_data_path="${PAR_PATH}/guest_test.csv"