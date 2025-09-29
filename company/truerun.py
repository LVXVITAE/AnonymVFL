import os
import warnings

# 在导入 JAX 之前设置
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=1 --xla_cpu_multi_thread_eigen=false'
os.environ['OMP_NUM_THREADS'] = '1'

# 过滤特定警告
warnings.filterwarnings('ignore', category=RuntimeWarning, module='subprocess')
# 屏蔽所有警告
warnings.filterwarnings('ignore')


from common import MPCInitializer
from secretflow.data.ndarray import load
import secretflow as sf
import pandas as pd
import numpy as np
import argparse

def main(args : argparse.Namespace):
    cluster_def = {}
    if args.mode == 'multi_sim' or args.mode == 'multi_distributed':
        company_spu_ip, company_spu_port = args.company_spu_addr.split(':')
        partner_spu_ip, partner_spu_port = args.partner_spu_addr.split(':')
        coordinator_spu_ip, coordinator_spu_port = args.coordinator_spu_addr.split(':')
        # 添加监听端口信息输出
        print("=" * 60)
        print("🚀 多方安全计算节点启动信息")
        print("=" * 60)
        print(f"📍 Company 节点监听地址:    {args.company_spu_addr}")
        print(f"📍 Partner 节点监听地址:    {args.partner_spu_addr}")
        print(f"📍 Coordinator 节点监听地址: {args.coordinator_spu_addr}")
        print(f"🌐 Ray Head 地址:          {args.ray_head_addr}")
        print("=" * 60)
        
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
        cluster_def['runtime_config'] = {
            'protocol': 3,
            'field': 3
        }
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
        print("\n" + "=" * 60)
        print("🔒 开始执行私有集合求交 (PSI)")
        print("=" * 60)
        print(f"📂 Company 训练数据: {args.path_to_company_train_dataset}")
        print(f"📂 Partner 训练数据:  {args.path_to_partner_train_dataset}")
        
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
        if 'Revenue' in company_header:
            train_label_keeper = company
        elif 'Revenue' in partner_header:
            train_label_keeper = partner
        else:
            raise ValueError("Label column 'Revenue' must be present in either company or partner dataset")
        #记录标签列的索引
        y_col = header.index('Revenue')

        from PSI import private_set_intersection
        company_share, partner_share, bucket_labels = private_set_intersection(company_data, partner_data,heu_devices)
        
        print(f"\n💾 保存加法秘密共享分片...")
        print(f"📁 Company 秘密共享分片: {args.path_to_company_share}")
        print(f"📁 Partner 秘密共享分片:  {args.path_to_partner_share}")
        company_share.device(np.savetxt)(args.path_to_company_share,company_share, delimiter=',')
        partner_share.device(np.savetxt)(args.path_to_partner_share,partner_share, delimiter=',')
    
        print(f"\n🔍 显示前10行数据")
        print("-" * 80)
        
        # # 获取明文数据进行对比 - 修正：处理元组格式的数据
        # company_data_revealed = sf.reveal(company_data)
        # partner_data_revealed = sf.reveal(partner_data)

        # # 提取实际的特征数据（第二个元素是private_features）
        # company_features = company_data_revealed[1][:10]  # private_features
        # partner_features = partner_data_revealed[1][:10]  # private_features

        company_share_revealed = sf.reveal(company_share)[:10]
        partner_share_revealed = sf.reveal(partner_share)[:10]

        # print("📊 Company 原始数据 (前10行前5列):")
        # print(company_features[:, :5])
        print(f"📊 Company 秘密共享分片 (前10行前5列):")
        print(company_share_revealed[:, :5])

        # print("\n📊 Partner 原始数据 (前10行前5列):")
        # print(partner_features[:, :5])
        print(f"📊 Partner 秘密共享分片 (前10行前5列):")
        print(partner_share_revealed[:, :5])
        
        # print("🔍 最原始的输入数据:")
        # print("Company原始数据前5行:", sf.reveal(company_data)[1][:5])
        # print("Partner原始数据前5行:", sf.reveal(partner_data)[1][:5])
        
        #  # 验证加法秘密共享的正确性
        # print(f"\n✅ 秘密共享验证:")
        # reconstructed = company_share_revealed + partner_share_revealed
        # print(f"📊 重构数据 (Company分片 + Partner分片，前10行前5列):")
        # print(reconstructed[:, :5])
        
        # # 计算原始数据的拼接（用于验证）
        # original_combined = np.hstack([company_data_revealed, partner_data_revealed])
        # print(f"📊 原始数据拼接 (Company + Partner，前10行前5列):")
        # print(original_combined[:, :5])
        
        # # 检查是否相等
        # max_diff = np.max(np.abs(reconstructed - original_combined))
        # print(f"🔍 重构误差 (最大绝对差值): {max_diff:.6f}")
        # if max_diff < 1e-6:
        #     print("✅ 秘密共享重构成功！")
        # else:
        #     print("❌ 秘密共享重构存在误差！")
        
        print("=" * 60)
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
        if 'Revenue' in test_company_header:
            y_col= test_company_header.index('Revenue')
            X_test_company, y_test = company(split_X_y,num_returns=2)(test_company,y_col)
            X_test_partner = test_partner
        elif 'Revenue' in test_partner_header:
            y_col= test_partner_header.index('Revenue')
            X_test_partner, y_test = partner(split_X_y,num_returns=2)(test_partner,y_col)
            X_test_company = test_company
        else:
            raise ValueError("Label column 'Revenue' must be present in either company or partner validation dataset")
        X_test = load({company: X_test_company, partner: X_test_partner})        
    else:
        X_test = None
        y_test = None

    from LR import SSLR
    
    print(f"\n🤖 开始训练联邦逻辑回归模型")
    print("=" * 60)
    print(f"📈 训练参数:")
    print(f"   • 训练轮数: {args.n_epochs}")
    print(f"   • 批次大小: {args.batch_size}")
    print(f"   • 学习率:   {args.lr}")
    print(f"   • 验证频率: 每 {args.val_steps} 步")
    print(f"   • 标签共享: {'是' if args.share_y else '否'}")
    print("=" * 60)

    model = SSLR(devices, approx=args.share_y)
    accs = model.fit(X_train, y_train, X_test, y_test, n_epochs=args.n_epochs, batch_size=args.batch_size, val_steps=args.val_steps, lr=args.lr)
    if hasattr(args, 'path_to_company_model_save_dir') and hasattr(args, 'path_to_partner_model_save_dir'):
        print(f"\n💾 保存训练好的模型权重")
        print("=" * 60)
        print(f"📁 Company 模型权重保存路径: {args.path_to_company_model_save_dir}")
        print(f"📁 Partner 模型权重保存路径:  {args.path_to_partner_model_save_dir}")
        model.save({
            'company': args.path_to_company_model_save_dir,
            'partner': args.path_to_partner_model_save_dir,
        },ext='csv')
        
        print(f"✅ 模型权重已成功保存！")
        print(f"📊 权重文件:")
        print(f"   • {args.path_to_company_model_save_dir}/weight.csv")
        print(f"   • {args.path_to_company_model_save_dir}/info.json")
        print(f"   • {args.path_to_partner_model_save_dir}/weight.csv")
        print(f"   • {args.path_to_partner_model_save_dir}/info.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="AnonymVFL",description="运行匿踪对齐和匿踪学习")
    parser.add_argument('--mode', type=str, default='single_sim', choices=['single_sim', 'multi_sim', 'multi_distributed'], help='运行模式，single_sim用于单机运行，multi_sim用于局域网多机运行，multi_distributed用于真正分布式运行')
    parser.add_argument('--party', type=str, choices=['company', 'partner', 'coordinator'], help='在multi_distributed模式下指定当前节点的角色')
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