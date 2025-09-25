from common import MPCInitializer
import secretflow as sf
import pandas as pd
import numpy as np
import argparse
from secretflow.data.ndarray import load
import os

def main(args : argparse.Namespace):
    cluster_def = {
        'runtime_config' : {
            'protocol': 3, # 3: ABY3, 5: SecureNN
            'field': 3
        }
    }
    if args.mode == 'multi_sim':
        company_spu_ip, company_spu_port = args.company_spu_addr.split(':')
        partner_spu_ip, partner_spu_port = args.partner_spu_addr.split(':')
        coordinator_spu_ip, coordinator_spu_port = args.coordinator_spu_addr.split(':')
        cluster_def['nodes'] = [
            {
                'party': 'company',
                # Please choose an unused port.
                'address': args.company_spu_addr,
                'listen_addr': f'0.0.0.0:{company_spu_port}'
            },
            {
                'party': 'partner',
                # Please choose an unused port.
                'address': args.partner_spu_addr,
                'listen_addr': f'0.0.0.0:{partner_spu_port}'
            },
            {
                'party': 'coordinator',
                # Please choose an unused port.
                'address': args.coordinator_spu_addr,
                'listen_addr': f'0.0.0.0:{coordinator_spu_port}'
            }
        ]
    mpc_init = MPCInitializer(args.mode, args.ray_head_addr,cluster_def)
    company, partner, coordinator = mpc_init.company, mpc_init.partner, mpc_init.coordinator
    spu = mpc_init.spu

    # 设置设备
    devices = {
        'spu': spu,
        'company': company,
        'partner': partner,
        'coordinator': coordinator,
    }

    if args.run_psi:
        heu_devices = (mpc_init.company_heu, mpc_init.partner_heu)

        if args.model == "SSLR":
            def read_dataset_lr(path: str):
                data = pd.read_csv(path)
                keys = data.iloc[:, 0].astype(str).tolist()
                private_features = data.iloc[:, 1:].to_numpy(dtype=np.float32)
                header = data.columns.tolist()
                return (keys, private_features,None), header

            company_data, company_header = company(read_dataset_lr,num_returns=2)(args.path_to_company_train_dataset)
            partner_data, partner_header = partner(read_dataset_lr,num_returns=2)(args.path_to_partner_train_dataset)
            #记录标签列的索引
            company_header = sf.reveal(company_header)
            company_header.remove('id')
            partner_header = sf.reveal(partner_header)
            partner_header.remove("id")
            # active_party一般为标签的持有者
            if 'y' in company_header:
                devices['active_party'] = company
            elif 'y' in partner_header:
                devices['active_party'] = partner
            else:
                raise ValueError("训练集标签列'y'未找到")
            
            train_label_keeper = devices['active_party']
            #输出交集的所在列的索引
            header = company_header + partner_header
            y_col = header.index('y')

        elif args.model == "SSXGBoost":
            def read_dataset_xgb(path: str):
                data = pd.read_csv(path)
                keys = data['id'].astype(str).tolist()
                private_features = data.drop(columns=['id']).to_numpy(dtype=np.float32)
                header = data.columns.tolist()
                return keys, private_features, header
            company_keys, company_features, company_header = company(read_dataset_xgb,num_returns=3)(args.path_to_company_train_dataset)
            partner_keys, partner_features, partner_header = partner(read_dataset_xgb,num_returns=3)(args.path_to_partner_train_dataset)

            company_train_features = company_features
            partner_train_features = partner_features

            #记录标签列的索引
            company_header = sf.reveal(company_header)
            company_header.remove('id')
            partner_header = sf.reveal(partner_header)
            partner_header.remove("id")
            # active_party一般为标签的持有者
            if 'y' in company_header:
                devices['active_party'] = company
                company_y_col = company_header.index('y')
                company_train_features = company(lambda features, y_col: np.delete(features, y_col, axis=1))(company_features, company_y_col)
            elif 'y' in partner_header:
                devices['active_party'] = partner
                partner_y_col = partner_header.index('y')
                partner_train_features = partner(lambda features, y_col: np.delete(features, y_col, axis=1))(partner_features, partner_y_col)
            else:
                raise ValueError("训练集标签列'y'未找到")
            
            train_label_keeper = devices['active_party']
            #输出交集的所在列的索引
            header = company_header + partner_header
            y_col = header.index('y')

            from XGBoost import quantize_buckets
            K_quantiles = args.K_quantiles
            Quantiles1, _, buckets_labels1 = company(quantize_buckets,num_returns=3)(company_train_features, k=K_quantiles)
            Quantiles2, _, buckets_labels2 = partner(quantize_buckets,num_returns=3)(partner_train_features, k=K_quantiles) #最后一列是y无需分桶

            # 分位点合并为联邦数组形式
            from secretflow.data import PartitionWay
            FedQuantiles = load({company: Quantiles1, partner: Quantiles2}, partition_way=PartitionWay.HORIZONTAL)
            # 将key,feature和bucket_label整合到一个元组中
            def make_triplet(x,y,z):
                return (x,y,z)

            company_data = company(make_triplet)(company_keys,company_features,buckets_labels1)
            partner_data = partner(make_triplet)(partner_keys,partner_features,buckets_labels2)

        from PSI import private_set_intersection
        company_share, partner_share, bucket_labels = private_set_intersection(company_data, partner_data,heu_devices)

        if args.model == "SSXGBoost":
            from XGBoost import recover_buckets
            buckets = recover_buckets(bucket_labels)
        
        if args.path_to_company_share is not None:
            company_share.device(np.savetxt)(args.path_to_company_share,company_share, delimiter=',')
        if args.path_to_partner_share is not None:
            partner_share.device(np.savetxt)(args.path_to_partner_share,partner_share, delimiter=',')
        if args.model == "SSXGBoost" and args.path_to_buckets is not None:
            np.savetxt(args.path_to_buckets,buckets, delimiter=',')
    else:
        if args.path_to_company_share is None or args.path_to_partner_share is None:
            raise ValueError("Please specify path to company share and partner share")
        if args.path_to_company_share.endswith('.csv'):
            company_share = company(np.loadtxt)(args.path_to_company_share, delimiter=',')
        elif args.path_to_company_share.endswith('.npy'):
            company_share = company(np.load)(args.path_to_company_share)
        else:
            raise ValueError("Unsupported file format for company share. Use .csv or .npy")
        
        if args.path_to_partner_share.endswith('.csv'):
            partner_share = partner(np.loadtxt)(args.path_to_partner_share, delimiter=',')
        elif args.path_to_partner_share.endswith('.npy'):
            partner_share = partner(np.load)(args.path_to_partner_share)
        else:
            raise ValueError("Unsupported file format for partner share. Use .csv or .npy")
        
        if args.model == "SSXGBoost" and args.path_to_buckets is None:
            raise ValueError("Please specify path to buckets")
        if args.path_to_buckets.endswith('.csv'):
            buckets = np.loadtxt(args.path_to_buckets, delimiter=',')
        elif args.path_to_buckets.endswith('.npy'):
            buckets = np.load(args.path_to_buckets)
        else:
            raise ValueError("Unsupported file format for buckets. Use .csv or .npy")

    # 将训练集转移到对应设备
    def split_X_y(share, y_col):
        X = np.delete(share, y_col, axis=1)
        y = share[:, y_col].reshape(-1, 1)
        return X, y

    def share2spu(company_share, partner_share):
        company_share = company_share.to(spu)
        partner_share = partner_share.to(spu)
        return spu(lambda x, y : x + y)(company_share, partner_share)

    X_train_company, y_train_company = company(split_X_y,num_returns=2)(company_share,y_col)
    X_train_partner, y_train_partner = partner(split_X_y,num_returns=2)(partner_share,y_col)
    X_train = share2spu(X_train_company, X_train_partner)
    if args.share_y:
        y_train = share2spu(y_train_company, y_train_partner)
    else:
        y_train = train_label_keeper(lambda x, y : x + y)(y_train_company.to(train_label_keeper), y_train_partner.to(train_label_keeper))

    # 读取验证集。验证集应为对齐好的纵向划分的明文联邦数据。
    def read_val_dataset(path):
        data = pd.read_csv(path)
        features = data.iloc[:, 1:].to_numpy(dtype=np.float32)
        header = data.columns.tolist()
        return features, header
    if args.path_to_company_val_dataset is not None and args.path_to_partner_val_dataset is not None:
        test_company, test_company_header = company(read_val_dataset, num_returns=2)(args.path_to_company_val_dataset)
        test_partner, test_partner_header = partner(read_val_dataset, num_returns=2)(args.path_to_partner_val_dataset)
        test_company_header = sf.reveal(test_company_header)[1:]
        test_partner_header = sf.reveal(test_partner_header)[1:]
        if 'y' in test_company_header:
            y_col= test_company_header.index('y')
            X_test_company, y_test = company(split_X_y,num_returns=2)(test_company,y_col)
            X_test_partner = test_partner
        elif 'y' in test_partner_header:
            y_col= test_partner_header.index('y')
            X_test_partner, y_test = partner(split_X_y,num_returns=2)(test_partner,y_col)
            X_test_company = test_company
        else:
            raise ValueError("验证集标签列'y'未找到")
        X_test = load({company: X_test_company, partner: X_test_partner})        
    else:
        X_test = None
        y_test = None

    if args.model == "SSLR":
        from LR import SSLR

        model = SSLR(devices,args.reg_coef)
        accs = model.fit(X_train, y_train, X_test, y_test, n_epochs=args.n_epochs, batch_size=args.batch_size, val_steps=args.val_steps, lr=args.lr)
        if hasattr(args, 'path_to_company_model_save_dir') and hasattr(args, 'path_to_partner_model_save_dir'):
            model.save({
                'company': args.path_to_company_model_save_dir,
                'partner': args.path_to_partner_model_save_dir,
            },ext='csv')
    elif args.model == "SSXGBoost":
        from XGBoost import SSXGBoost
        model = SSXGBoost(devices=devices, n_estimators=args.n_estimators, lambda_=args.reg_coef, max_depth=args.max_depth)

        train_accs, test_accs = model.fit(X_train, y_train, buckets, FedQuantiles,X_test, y_test)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="AnonymVFL",description="运行匿踪对齐和匿踪学习")
    parser.add_argument('--mode', type=str, default='single_sim', choices=['single_sim', 'multi_sim'], help='运行模式，single_sim用于单机运行，multi_sim用于局域网多机运行')
    parser.add_argument('--ray_head_addr', type=str, default="", help='Ray集群的头节点地址')
    parser.add_argument('--company_spu_addr', type=str, required=False, help='Company SPU的地址，注意不要和Ray端口冲突')
    parser.add_argument('--partner_spu_addr', type=str, required=False, help='Partner SPU的地址，注意不要和Ray端口冲突')
    parser.add_argument('--coordinator_spu_addr', type=str, required=False, help='Coordinator SPU的地址，注意不要和Ray端口冲突')
    parser.add_argument('--run_psi', type=bool, default=True, help='是否运行PSI')
    parser.add_argument('--path_to_company_train_dataset', type=str, default="", help='Company端训练集路径。数据集应为明文csv文件。')
    parser.add_argument('--path_to_partner_train_dataset', type=str, default="", help='Partner端训练集路径。数据集应为明文csv文件。')
    parser.add_argument('--path_to_company_share', type=str, required=False, help='Company端PSI输出共享分片保存路径。如运行PSI，保存到该路径；如不运行PSI，从该路径读取分片')
    parser.add_argument('--path_to_partner_share', type=str, required=False, help='Partner端PSI输出共享分片保存路径。如运行PSI，保存到该路径；如不运行PSI，从该路径读取分片')
    parser.add_argument('--path_to_buckets', type=str, required=False, help='分桶信息保存路径。保存在host(即运行Python代码)的设备上。如运行PSI，保存到该路径；如不运行PSI，从该路径读取分片')
    parser.add_argument('--share_y', type=bool, default=False, help='是否秘密共享共享标签y')
    parser.add_argument('--path_to_company_val_dataset', type=str, default="", help='Company端验证集路径。数据集应为明文csv文件。')
    parser.add_argument('--path_to_partner_val_dataset', type=str, default="", help='Partner端验证集路径。数据集应为明文csv文件。')
    parser.add_argument('--model', type=str, default='SSLR', choices=['SSLR', 'SSXGBoost'], help='选择要运行的模型')
    parser.add_argument('--reg_coef', type=float, default=0.0, help='正则化系数')
    parser.add_argument('--n_epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=1024, help='批次大小')
    parser.add_argument('--val_steps', type=int, default=1, help='每隔多少步在验证集上评估一次')
    parser.add_argument('--lr', type=float, default=0.1, help='初始学习率')
    parser.add_argument('--K_quantiles', type=int, default=20, help='分桶的分位数个数')
    parser.add_argument('--n_estimators', type=int, default=2, help='XGBoost的决策树数量')
    parser.add_argument('--max_depth', type=int, default=2, help='XGBoost的树最大深度')
    parser.add_argument('--path_to_company_model_save_dir', type=str, required=False, help='Company端模型保存路径')
    parser.add_argument('--path_to_partner_model_save_dir', type=str, required=False, help='Partner端模型保存路径')
    args = parser.parse_args()
    main(args)
    os._exit(0)