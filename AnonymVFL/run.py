from common import MPCInitializer
import secretflow as sf
import pandas as pd
import numpy as np
import argparse
from secretflow.data.ndarray import load

def main(args : argparse.Namespace):
    cluster_def = {}
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
        cluster_def['runtime_config'] = {}
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

        def read_dataset(path: str):
            data = pd.read_csv(path)
            keys = data.iloc[:, 0].astype(str).tolist()
            private_features = data.iloc[:, 1:].to_numpy(dtype=np.float32)
            header = data.columns.tolist()
            return (keys, private_features,None), header

        company_data, company_header = company(read_dataset,num_returns=2)(args.path_to_company_train_dataset)
        partner_data, partner_header = partner(read_dataset,num_returns=2)(args.path_to_partner_train_dataset)
        company_header = sf.reveal(company_header)[1:]
        partner_header = sf.reveal(partner_header)[1:]
        #输出交集的列标题
        header = company_header + partner_header
        # 记录分隔列的索引。索引左侧列为company的特征，右侧列为partner的特征
        if 'y' in company_header:
            train_label_keeper = company
        elif 'y' in partner_header:
            train_label_keeper = partner
        else:
            raise ValueError("Label column 'y' must be present in either company or partner dataset")
        #记录标签列的索引
        y_col = header.index('y')

        from PSI import private_set_intersection
        company_share, partner_share, bucket_labels = private_set_intersection(company_data, partner_data,heu_devices)
        company_share.device(np.savetxt)(args.path_to_company_share,company_share, delimiter=',')
        partner_share.device(np.savetxt)(args.path_to_partner_share,partner_share, delimiter=',')
    else:
        if args.path_to_company_share.endswith('.csv'):
            company_share = company(np.loadtxt)(args.path_to_company_share_save, delimiter=',')
        elif args.path_to_company_share.endswith('.npy'):
            company_share = company(np.load)(args.path_to_company_share_save)
        else:
            raise ValueError("Unsupported file format for company share. Use .csv or .npy")
        if args.path_to_partner_share.endswith('.csv'):
            partner_share = partner(np.loadtxt)(args.path_to_partner_share, delimiter=',')
        elif args.path_to_partner_share.endswith('.npy'):
            partner_share = partner(np.load)(args.path_to_partner_share)
        else:
            raise ValueError("Unsupported file format for partner share. Use .csv or .npy")



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
        y_train = train_label_keeper(lambda x, y : x + y)(y_train_company.to(train_label_keeper), y_train_partner.to(train_label_keeper))
    else:
        y_train = share2spu(y_train_company, y_train_partner)

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
            raise ValueError("Label column 'y' must be present in either company or partner validation dataset")
        X_test = load({company: X_test_company, partner: X_test_partner})        
    else:
        X_test = None
        y_test = None

    from LR import SSLR
    import matplotlib.pyplot as plt

    model = SSLR(devices, approx=args.share_y)
    accs = model.fit(X_train, y_train, X_test, y_test, n_epochs=args.n_epochs, batch_size=args.batch_size, val_steps=args.val_steps, lr=args.lr)
    if hasattr(args, 'path_to_company_model_save_dir') and hasattr(args, 'path_to_partner_model_save_dir'):
        model.save({
            'company': args.path_to_company_model_save_dir,
            'partner': args.path_to_partner_model_save_dir,
        },ext='csv')
    plt.plot(accs,label = "SSLR",color = "blue")
    company(plt.savefig)(args.path_to_company_model_save_dir + '/accuracy_curve.png')
    partner(plt.savefig)(args.path_to_partner_model_save_dir + '/accuracy_curve.png')

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
    parser.add_argument('--path_to_company_share', type=str, default="", help='Company端PSI输出共享分片保存路径。如运行PSI，保存到该路径；如不运行PSI，从该路径读取分片')
    parser.add_argument('--path_to_partner_share', type=str, default="", help='Partner端PSI输出共享分片保存路径。如运行PSI，保存到该路径；如不运行PSI，从该路径读取分片')
    parser.add_argument('--share_y', type=bool, default=False, help='是否秘密共享共享标签y')
    parser.add_argument('--path_to_company_val_dataset', type=str, default="", help='Company端验证集路径。数据集应为明文csv文件。')
    parser.add_argument('--path_to_partner_val_dataset', type=str, default="", help='Partner端验证集路径。数据集应为明文csv文件。')
    parser.add_argument('--model', type=str, default='SSLR', choices=['SSLR', 'SSXGBoost'], help='选择要运行的模型')
    parser.add_argument('--n_epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=1024, help='批次大小')
    parser.add_argument('--val_steps', type=int, default=1, help='每隔多少步在验证集上评估一次')
    parser.add_argument('--lr', type=float, default=0.1, help='初始学习率')
    parser.add_argument('--path_to_company_model_save_dir', type=str, required=False, help='Company端模型保存路径')
    parser.add_argument('--path_to_partner_model_save_dir', type=str, required=False, help='Partner端模型保存路径')
    args = parser.parse_args()
    main(args)