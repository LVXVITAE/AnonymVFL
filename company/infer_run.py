"""
推理入口脚本（子进程方式执行），仿照 truerun.py 的分布式初始化方式。
app.py 通过 subprocess 调用本脚本，并解析 stdout 中最后一行 JSON 获取结果。
"""
import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

# 在导入 JAX 之前设置
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=1 --xla_cpu_multi_thread_eigen=false'
os.environ['OMP_NUM_THREADS'] = '1'

warnings.filterwarnings('ignore')


def main(args: argparse.Namespace):
    import secretflow as sf
    from common import MPCInitializer
    from infer import InferEngine

    cluster_def = {}
    if args.mode == 'multi_distributed':
        company_spu_ip, company_spu_port = args.company_spu_addr.split(':')
        partner_spu_ip, partner_spu_port = args.partner_spu_addr.split(':')
        coordinator_spu_ip, coordinator_spu_port = args.coordinator_spu_addr.split(':')

        cluster_def['nodes'] = [
            {
                'party': 'company',
                'address': args.company_spu_addr,
                'listen_addr': f'0.0.0.0:{company_spu_port}'
            },
            {
                'party': 'partner',
                'address': args.partner_spu_addr,
                'listen_addr': f'0.0.0.0:{partner_spu_port}'
            },
            {
                'party': 'coordinator',
                'address': args.coordinator_spu_addr,
                'listen_addr': f'0.0.0.0:{coordinator_spu_port}'
            }
        ]
        cluster_def['runtime_config'] = {
            'protocol': 3,
            'field': 3
        }

    mpc_init = MPCInitializer(args.mode, args.ray_head_addr, cluster_def)
    company = mpc_init.company
    partner = mpc_init.partner
    coordinator = mpc_init.coordinator

    devices = {
        'company': company,
        'partner': partner,
        'coordinator': coordinator,
    }

    print(f"[infer_run] 加载模型: {args.model}", file=sys.stderr)
    engine = InferEngine(devices, model=args.model)
    engine.load_model({
        'company': args.company_model_path,
        'partner': args.partner_model_path,
    })

    print("[infer_run] 加载推理数据...", file=sys.stderr)
    company_keys, company_data, partner_keys, partner_data = engine.load_data_from_path({
        'company': args.company_data_path,
        'partner': args.partner_data_path,
    })

    print("[infer_run] 计算 PSI 交集...", file=sys.stderr)
    engine.compute_intersection(company_keys, company_data, partner_keys, partner_data)

    print("[infer_run] 执行推理...", file=sys.stderr)
    pred = engine.infer(device=company)
    pred_df = sf.reveal(pred)

    # 输出结果 JSON（最后一行，供 app.py 解析）
    result = json.loads(pred_df.to_json(orient='records'))
    print(json.dumps({'status': 'success', 'result': result}), flush=True)

    sf.shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AnonymVFL 推理脚本')
    parser.add_argument('--mode', type=str, default='multi_distributed',
                        choices=['single_sim', 'multi_sim', 'multi_distributed'])
    parser.add_argument('--ray_head_addr', type=str, default='')
    parser.add_argument('--company_spu_addr', type=str, default='')
    parser.add_argument('--partner_spu_addr', type=str, default='')
    parser.add_argument('--coordinator_spu_addr', type=str, default='')
    parser.add_argument('--model', type=str, default='SSLR',
                        choices=['SSLR', 'SSXGBoost'])
    parser.add_argument('--company_model_path', type=str, required=True)
    parser.add_argument('--partner_model_path', type=str, required=True)
    parser.add_argument('--company_data_path', type=str, required=True)
    parser.add_argument('--partner_data_path', type=str, required=True)
    args = parser.parse_args()
    main(args)
