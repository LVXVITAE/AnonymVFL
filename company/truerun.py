# 导入命令行参数解析和数值计算相关库
import argparse
import numpy as np
import pandas as pd
import secretflow as sf
from secretflow.data.ndarray import load
from secretflow.data import PartitionWay
from common import MPCInitializer
import os
import warnings

# 在导入 JAX 之前设置环境变量，强制使用CPU并限制线程数
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=1 --xla_cpu_multi_thread_eigen=false'
os.environ['OMP_NUM_THREADS'] = '1'

# 过滤特定警告
warnings.filterwarnings('ignore', category=RuntimeWarning, module='subprocess')
# 屏蔽所有警告
warnings.filterwarnings('ignore')


def main(args: argparse.Namespace):
    """主函数，根据命令行参数执行纵向联邦学习训练流程
    
    Args:
        args: 解析后的命令行参数对象
    """
    # 初始化集群配置字典，用于存储SPU节点信息
    cluster_def = {}
    # 判断是否为多机模式（模拟或真实分布式），需要配置集群信息
    if args.mode == 'multi_sim' or args.mode == 'multi_distributed':
        # 解析Company、Partner和Coordinator端SPU的IP地址和端口号
        company_spu_ip, company_spu_port = args.company_spu_addr.split(':')
        partner_spu_ip, partner_spu_port = args.partner_spu_addr.split(':')
        coordinator_spu_ip, coordinator_spu_port = args.coordinator_spu_addr.split(
            ':')
        # 添加监听端口信息输出，方便用户确认配置是否正确
        print("=" * 60)
        print("🚀 多方安全计算节点启动信息")
        print("=" * 60)
        print(f"📍 Company 节点监听地址:    {args.company_spu_addr}")
        print(f"📍 Partner 节点监听地址:    {args.partner_spu_addr}")
        print(f"📍 Coordinator 节点监听地址: {args.coordinator_spu_addr}")
        print(f"🌐 Ray Head 地址:          {args.ray_head_addr}")
        print("=" * 60)

        # 配置集群节点列表，包含三个参与方的SPU信息
        cluster_def['nodes'] = [
            {
                'party': 'company',
                # 请选择一个未被占用的端口
                'address': args.company_spu_addr,
                'listen_addr': f'0.0.0.0:{company_spu_port}'
            },
            {
                'party': 'partner',
                # 请选择一个未被占用的端口
                'address': args.partner_spu_addr,
                'listen_addr': f'0.0.0.0:{partner_spu_port}'
            },
            {
                'party': 'coordinator',
                # 请选择一个未被占用的端口
                'address': args.coordinator_spu_addr,
                'listen_addr': f'0.0.0.0:{coordinator_spu_port}'
            }
        ]
        # 配置SPU运行时参数
        cluster_def['runtime_config'] = {
            'protocol': 3,
            'field': 3
        }
    # 初始化MPC环境，创建各方设备和SPU
    mpc_init = MPCInitializer(args.mode, args.ray_head_addr, cluster_def)
    company, partner, coordinator = mpc_init.company, mpc_init.partner, mpc_init.coordinator
    spu = mpc_init.spu

    # 设置设备
    devices = {
        'spu': spu,
        'company': company,
        'partner': partner,
        'coordinator': coordinator,
    }

    train_label_keeper = company
    y_col = None
    FedQuantiles = None
    buckets = None

    if args.run_psi:
        print("\n" + "=" * 60)
        print("🔒 开始执行私有集合求交 (PSI)")
        print("=" * 60)
        print(f"📂 Company 训练数据: {args.path_to_company_train_dataset}")
        print(f"📂 Partner 训练数据:  {args.path_to_partner_train_dataset}")

        # 获取HEU（同态加密单元）设备元组，用于加密计算
        heu_devices = (mpc_init.company_heu, mpc_init.partner_heu)

        if args.model == "SSLR":
            def read_dataset(path: str):
                """读取CSV数据集并提取键和特征
                """
                # 读取CSV文件到DataFrame
                data = pd.read_csv(path)
                # 提取第一列作为ID键，并转换为字符串列表
                keys = data.iloc[:, 0].astype(str).tolist()
                # 提取除第一列外的所有列作为特征，转换为float32类型
                private_features = data.iloc[:, 1:].to_numpy(dtype=np.float32)
                # 获取列名列表作为表头
                header = data.columns.tolist()
                return (keys, private_features, None), header

            # 读取训练数据
            company_data, company_header = company(
                read_dataset, num_returns=2)(args.path_to_company_train_dataset)
            partner_data, partner_header = partner(
                read_dataset, num_returns=2)(args.path_to_partner_train_dataset)
            # 揭示表头信息，去除第一列（ID列）
            company_header = sf.reveal(company_header)[1:]
            partner_header = sf.reveal(partner_header)[1:]
            # 合并双方的特征列名，形成完整的特征列表
            header = company_header + partner_header

            # 检查标签列'Revenue'位置，分离特征和标签
            if 'Revenue' in company_header:
                train_label_keeper = company
            elif 'Revenue' in partner_header:
                train_label_keeper = partner
            else:
                # 如果双方都没有标签列，抛出异常
                raise ValueError(
                    "Label column 'Revenue' must be present in either company or partner dataset")
            # 将活跃方（标签持有者）添加到设备字典中
            devices['active_party'] = train_label_keeper
            # 获取标签列在合并后特征列表中的索引位置
            y_col = header.index('Revenue')

        elif args.model == "SSXGBoost":
            def read_dataset_xgb(path: str):
                """读取CSV数据集，专门用于XGBoost模型
                """
                # 读取CSV文件到DataFrame
                data = pd.read_csv(path)
                # 提取'id'列作为键，转换为字符串列表
                keys = data['id'].astype(str).tolist()
                # 删除'id'列后，将剩余列转换为float32类型的数组
                private_features = data.drop(
                    columns=['id']).to_numpy(dtype=np.float32)
                # 获取所有列名作为表头
                header = data.columns.tolist()
                return keys, private_features, header

            # 读取训练数据
            company_keys, company_features, company_header = company(
                read_dataset_xgb, num_returns=3)(args.path_to_company_train_dataset)
            partner_keys, partner_features, partner_header = partner(
                read_dataset_xgb, num_returns=3)(args.path_to_partner_train_dataset)

            company_train_features = company_features
            partner_train_features = partner_features

            # 揭示表头并移除'id'列
            company_header = sf.reveal(company_header)
            company_header.remove('id')
            partner_header = sf.reveal(partner_header)
            partner_header.remove('id')

            # 检查标签列位置，分离特征和标签
            if 'Revenue' in company_header:
                train_label_keeper = company
                company_y_col = company_header.index('Revenue')
                company_train_features = company(lambda features, y_idx: np.delete(
                    features, y_idx, axis=1))(company_features, company_y_col)
            elif 'Revenue' in partner_header:
                train_label_keeper = partner
                partner_y_col = partner_header.index('Revenue')
                partner_train_features = partner(lambda features, y_idx: np.delete(
                    features, y_idx, axis=1))(partner_features, partner_y_col)
            else:
                # 如果双方都没有标签列，抛出异常
                raise ValueError(
                    "Label column 'Revenue' must be present in either company or partner dataset")
            # 将活跃方添加到设备字典中
            devices['active_party'] = train_label_keeper

            # 合并双方的特征列名
            header = company_header + partner_header
            # 获取标签列在合并后特征列表中的索引
            y_col = header.index('Revenue')

            from XGBoost import quantize_buckets
            # 获取量化分位数参数
            K_quantiles = args.K_quantiles

            if train_label_keeper == company:
                Quantiles1, _, buckets_labels1 = company(quantize_buckets, num_returns=3)(
                    company_train_features, k=K_quantiles)
                Quantiles2, _, buckets_labels2 = partner(
                    quantize_buckets, num_returns=3)(partner_features, k=K_quantiles)
            else:
                Quantiles1, _, buckets_labels1 = company(
                    quantize_buckets, num_returns=3)(company_features, k=K_quantiles)
                Quantiles2, _, buckets_labels2 = partner(quantize_buckets, num_returns=3)(
                    partner_train_features, k=K_quantiles)

            # 将双方的量化结果水平合并，形成联邦量化数据
            FedQuantiles = load(
                {company: Quantiles1, partner: Quantiles2}, partition_way=PartitionWay.HORIZONTAL)

            def make_triplet(x, y, z):
                """将三个参数组合成元组"""
                return (x, y, z)

            if train_label_keeper == company:
                company_data = company(make_triplet)(
                    company_keys, company_train_features, buckets_labels1)
                partner_data = partner(make_triplet)(
                    partner_keys, partner_features, buckets_labels2)
            else:
                company_data = company(make_triplet)(
                    company_keys, company_features, buckets_labels1)
                partner_data = partner(make_triplet)(
                    partner_keys, partner_train_features, buckets_labels2)
        else:
            # 如果模型类型不支持，抛出异常
            raise ValueError(f"Unsupported model type: {args.model}")

        from PSI import private_set_intersection
        # 执行PSI操作，获取双方的秘密共享分片和分桶标签
        company_share, partner_share, bucket_labels = private_set_intersection(
            company_data, partner_data, heu_devices)

        # 如果模型是SSXGBoost，需要处理分桶信息
        if args.model == "SSXGBoost":
            from XGBoost import recover_buckets
            # 从分桶标签恢复完整的分桶信息
            buckets = recover_buckets(bucket_labels)
            # 如果没有指定分桶保存路径，使用默认路径（保存在Company分片同级目录下的buckets.npy）
            if not args.path_to_buckets:
                default_bucket_path = os.path.join(os.path.dirname(
                    args.path_to_company_share), "buckets.npy")
                args.path_to_buckets = default_bucket_path
            bucket_path = args.path_to_buckets if args.path_to_buckets.endswith(
                '.npy') else f"{args.path_to_buckets}.npy"
            # 保存分桶信息到文件
            np.save(bucket_path, buckets, allow_pickle=True)
            args.path_to_buckets = bucket_path

        print(f"\n💾 保存加法秘密共享分片...")
        print(f"📁 Company 秘密共享分片: {args.path_to_company_share}")
        print(f"📁 Partner 秘密共享分片:  {args.path_to_partner_share}")
        # 在双方设备上保存分片到CSV文件
        company_share.device(np.savetxt)(
            args.path_to_company_share, company_share, delimiter=',')
        partner_share.device(np.savetxt)(
            args.path_to_partner_share, partner_share, delimiter=',')

        print(f"\n🔍 显示前10行数据")
        print("-" * 80)

        # 获取明文数据进行对比 - 修正：处理元组格式的数据
        # company_data_revealed = sf.reveal(company_data)
        # partner_data_revealed = sf.reveal(partner_data)

        # # 提取实际的特征数据（第二个元素是private_features）
        # # private_features 前10行前5列
        # company_features = company_data_revealed[1][:10, :5]
        # # private_features 前10行前5列
        # partner_features = partner_data_revealed[1][:10, :5]

        # 揭示双方分片的前10行数据
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

        print("🔍 最原始的输入数据:")
        print("Company原始数据前10行前5列:", sf.reveal(company_data)[1][:10, :5])
        print("Partner原始数据前10行前5列:", sf.reveal(partner_data)[1][:10, :5])

        # # 验证加法秘密共享的正确性
        # print(f"\n✅ 秘密共享验证:")
        # reconstructed = company_share_revealed + partner_share_revealed
        # print(f"📊 重构数据 (Company分片 + Partner分片，前10行前5列):")
        # print(reconstructed[:, :5])

        # # 注意：这里只取前10行前5列进行验证
        # original_combined = np.hstack([company_features, partner_features])
        # print(f"📊 原始数据拼接 (Company + Partner，前10行前5列):")
        # print(original_combined[:, :5])

        # # 检查是否相等
        # max_diff = np.max(
        #     np.abs(reconstructed[:, :5] - original_combined[:, :5]))
        # print(f"🔍 重构误差 (最大绝对差值): {max_diff:.6f}")
        # if max_diff < 1e-3:
        #     print("✅ 秘密共享重构成功！")
        # else:
        #     print("❌ 秘密共享重构存在误差！")
        print("=" * 60)
    else:
        # 如果不运行PSI，需要从文件加载已有的秘密共享分片
        if not args.path_to_company_share or not args.path_to_partner_share:
            raise ValueError(
                "Please specify path to company and partner shares when PSI is disabled")
        if args.path_to_company_share.endswith('.csv'):
            # 从CSV文件加载Company端分片
            company_share = company(np.loadtxt)(
                args.path_to_company_share, delimiter=',')
        elif args.path_to_company_share.endswith('.npy'):
            # 从NPY文件加载Company端分片
            company_share = company(np.load)(args.path_to_company_share)
        else:
            raise ValueError(
                "Unsupported file format for company share. Use .csv or .npy")
        if args.path_to_partner_share.endswith('.csv'):
            # 从CSV文件加载Partner端分片
            partner_share = partner(np.loadtxt)(
                args.path_to_partner_share, delimiter=',')
        elif args.path_to_partner_share.endswith('.npy'):
            # 从NPY文件加载Partner端分片
            partner_share = partner(np.load)(args.path_to_partner_share)
        else:
            # 不支持的文件格式，抛出异常
            raise ValueError(
                "Unsupported file format for partner share. Use .csv or .npy")

        # 如果模型是SSXGBoost，还需要加载分桶信息
        if args.model == "SSXGBoost":
            # 检查是否指定了分桶文件路径
            if not args.path_to_buckets:
                raise ValueError(
                    "Please specify path to buckets when running SSXGBoost without PSI")
            bucket_path = args.path_to_buckets if args.path_to_buckets.endswith(
                '.npy') else f"{args.path_to_buckets}.npy"
            # 检查分桶文件是否存在
            if not os.path.exists(bucket_path):
                raise FileNotFoundError(
                    f"Buckets file not found: {bucket_path}")
            # 加载分桶信息
            buckets = np.load(bucket_path, allow_pickle=True)
            # 更新分桶路径参数
            args.path_to_buckets = bucket_path

    # 更新设备字典中的活跃方设置
    devices['active_party'] = train_label_keeper

    def split_X_y(share, y_col):
        """将数据分片分离为特征X和标签y
        """
        X = np.delete(share, y_col, axis=1)
        y = share[:, y_col].reshape(-1, 1)
        return X, y

    def share2spu(company_share, partner_share):
        """将Company和Partner的分片传输到SPU并求和
        """
        company_share = company_share.to(spu)
        partner_share = partner_share.to(spu)
        return spu(lambda x, y: x + y)(company_share, partner_share)

    # 双方分离特征和标签
    X_train_company, y_train_company = company(
        split_X_y, num_returns=2)(company_share, y_col)
    X_train_partner, y_train_partner = partner(
        split_X_y, num_returns=2)(partner_share, y_col)
    # 合并双方的特征数据到SPU
    X_train = share2spu(X_train_company, X_train_partner)
    if args.share_y:
        # 如果需要共享标签
        if args.model == "SSXGBoost":
            # XGBoost模型：在SPU上合并标签
            y_train = share2spu(y_train_company, y_train_partner)
        else:
            # 其他模型：在标签持有者设备上合并标签
            y_train = train_label_keeper(lambda x, y: x + y)(y_train_company.to(
                train_label_keeper), y_train_partner.to(train_label_keeper))
    else:
        # 如果不需要共享标签
        if args.model == "SSXGBoost":
            # XGBoost模型：在标签持有者设备上合并标签
            y_train = train_label_keeper(lambda x, y: x + y)(y_train_company.to(
                train_label_keeper), y_train_partner.to(train_label_keeper))
        else:
            # 其他模型：在SPU上合并标签
            y_train = share2spu(y_train_company, y_train_partner)

    def read_val_dataset(path):
        """读取验证数据集
        """
        data = pd.read_csv(path)
        # 提取除第一列外的所有列作为特征
        features = data.iloc[:, 1:].to_numpy(dtype=np.float32)
        # 获取列名列表
        header = data.columns.tolist()
        return features, header
    if args.path_to_company_val_dataset is not None and args.path_to_partner_val_dataset is not None:
        # 在双方设备上读取验证数据
        test_company, test_company_header = company(
            read_val_dataset, num_returns=2)(args.path_to_company_val_dataset)
        test_partner, test_partner_header = partner(
            read_val_dataset, num_returns=2)(args.path_to_partner_val_dataset)
        # 揭示验证数据的表头，去除第一列
        test_company_header = sf.reveal(test_company_header)[1:]
        test_partner_header = sf.reveal(test_partner_header)[1:]
        # 检查标签列位置，分离特征和标签
        if 'Revenue' in test_company_header:
            y_col = test_company_header.index('Revenue')
            X_test_company, y_test = company(
                split_X_y, num_returns=2)(test_company, y_col)
            X_test_partner = test_partner
        elif 'Revenue' in test_partner_header:
            y_col = test_partner_header.index('Revenue')
            X_test_partner, y_test = partner(
                split_X_y, num_returns=2)(test_partner, y_col)
            X_test_company = test_company
        else:
            # 如果双方都没有标签列，抛出异常
            raise ValueError(
                "Label column 'Revenue' must be present in either company or partner validation dataset")
        # 合并双方的验证特征数据
        X_test = load({company: X_test_company, partner: X_test_partner})
    else:
        # 如果没有提供验证数据集，设置为None
        X_test = None
        y_test = None

    if args.model == "SSLR":
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
        # 执行模型训练，返回每轮的准确率
        accs = model.fit(X_train, y_train, X_test, y_test, n_epochs=args.n_epochs,
                         batch_size=args.batch_size, val_steps=args.val_steps, lr=args.lr)
        # 检查是否指定了模型保存路径
        if hasattr(args, 'path_to_company_model_save_dir') and hasattr(args, 'path_to_partner_model_save_dir'):
            print(f"\n💾 保存训练好的模型权重")
            print("=" * 60)
            print(f"📁 Company 模型权重保存路径: {args.path_to_company_model_save_dir}")
            print(
                f"📁 Partner 模型权重保存路径:  {args.path_to_partner_model_save_dir}")
            # 保存
            model.save({
                'company': args.path_to_company_model_save_dir,
                'partner': args.path_to_partner_model_save_dir,
            }, ext='csv')

            print(f"✅ 模型权重已成功保存！")
            print(f"📊 权重文件:")
            print(f"   • {args.path_to_company_model_save_dir}/weight.csv")
            print(f"   • {args.path_to_company_model_save_dir}/info.json")
            print(f"   • {args.path_to_partner_model_save_dir}/weight.csv")
            print(f"   • {args.path_to_partner_model_save_dir}/info.json")
    elif args.model == "SSXGBoost":
        # 检查联邦分位数和分桶信息是否存在
        if FedQuantiles is None or buckets is None:
            raise ValueError(
                "FedQuantiles or buckets are missing for SSXGBoost training")
        from XGBoost import SSXGBoost

        print(f"\n🌲 开始训练联邦XGBoost模型")
        print("=" * 60)
        print(f"📈 训练参数:")
        print(f"   • 树数量: {args.n_estimators}")
        print(f"   • 最大深度: {args.max_depth}")
        print(f"   • 正则化系数: {args.reg_coef}")
        print(f"   • 分位点数: {args.K_quantiles}")
        print(f"   • 标签共享: {'是' if args.share_y else '否'}")
        print("=" * 60)

        model = SSXGBoost(devices=devices, n_estimators=args.n_estimators,
                          lambda_=args.reg_coef, max_depth=args.max_depth)
        # 执行模型训练，返回训练和测试准确率
        train_accs, test_accs = model.fit(
            X_train, y_train, buckets, FedQuantiles, X_test, y_test)
        # 输出结果
        if train_accs is not None:
            print(f"📊 训练集准确率: {train_accs}")
        if test_accs is not None:
            print(f"📊 测试集准确率: {test_accs}")

        # 保存模型
        if hasattr(args, 'path_to_company_model_save_dir') and hasattr(args, 'path_to_partner_model_save_dir') and args.path_to_company_model_save_dir and args.path_to_partner_model_save_dir:
            print(f"\n💾 保存训练好的模型")
            print("=" * 60)
            print(f"📁 Company 模型保存路径: {args.path_to_company_model_save_dir}")
            print(f"📁 Partner 模型保存路径:  {args.path_to_partner_model_save_dir}")
            try:
                # 尝试保存模型到指定目录
                model.save({
                    'company': args.path_to_company_model_save_dir,
                    'partner': args.path_to_partner_model_save_dir,
                }, ext='csv')
                print(f"✅ 模型已成功保存！")
                print(f"📊 保存的文件:")
                print(f"   • {args.path_to_company_model_save_dir}/weight.csv")
                print(
                    f"   • {args.path_to_company_model_save_dir}/quantiles.csv")
                print(f"   • {args.path_to_company_model_save_dir}/info.json")
                print(f"   • {args.path_to_company_model_save_dir}/tree.pkl")
                print(f"   • {args.path_to_partner_model_save_dir}/weight.csv")
                print(
                    f"   • {args.path_to_partner_model_save_dir}/quantiles.csv")
                print(f"   • {args.path_to_partner_model_save_dir}/info.json")
                print(f"   • {args.path_to_partner_model_save_dir}/tree.pkl")
            except Exception as e:
                # 如果保存失败，输出错误信息
                print(f"❌ 模型保存失败: {str(e)}")
                import traceback
                traceback.print_exc()
    else:
        # 如果模型类型不支持，抛出异常
        raise ValueError(f"Unsupported model type: {args.model}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="AnonymVFL", description="运行匿踪对齐和匿踪学习")
    parser.add_argument('--mode', type=str, default='single_sim', choices=[
                        'single_sim', 'multi_sim', 'multi_distributed'], help='运行模式，single_sim用于单机运行，multi_sim用于局域网多机运行，multi_distributed用于真正分布式运行')
    parser.add_argument('--party', type=str, choices=[
                        'company', 'partner', 'coordinator'], help='在multi_distributed模式下指定当前节点的角色')
    parser.add_argument('--ray_head_addr', type=str,
                        default="", help='Ray集群的头节点地址')
    parser.add_argument('--company_spu_addr', type=str,
                        required=False, help='Company SPU的地址，注意不要和Ray端口冲突')
    parser.add_argument('--partner_spu_addr', type=str,
                        required=False, help='Partner SPU的地址，注意不要和Ray端口冲突')
    parser.add_argument('--coordinator_spu_addr', type=str,
                        required=False, help='Coordinator SPU的地址，注意不要和Ray端口冲突')
    parser.add_argument('--run_psi', type=bool, default=True, help='是否运行PSI')
    parser.add_argument('--path_to_company_train_dataset',
                        type=str, default="", help='Company端训练集路径。数据集应为明文csv文件。')
    parser.add_argument('--path_to_partner_train_dataset',
                        type=str, default="", help='Partner端训练集路径。数据集应为明文csv文件。')
    parser.add_argument('--path_to_company_share', type=str, default="",
                        help='Company端PSI输出共享分片保存路径。如运行PSI，保存到该路径；如不运行PSI，从该路径读取分片')
    parser.add_argument('--path_to_partner_share', type=str, default="",
                        help='Partner端PSI输出共享分片保存路径。如运行PSI，保存到该路径；如不运行PSI，从该路径读取分片')
    parser.add_argument('--share_y', type=bool,
                        default=False, help='是否秘密共享共享标签y')
    parser.add_argument('--path_to_company_val_dataset',
                        type=str, default="", help='Company端验证集路径。数据集应为明文csv文件。')
    parser.add_argument('--path_to_partner_val_dataset',
                        type=str, default="", help='Partner端验证集路径。数据集应为明文csv文件。')
    parser.add_argument('--model', type=str, default='SSLR',
                        choices=['SSLR', 'SSXGBoost'], help='选择要运行的模型')
    parser.add_argument('--n_epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=1024, help='批次大小')
    parser.add_argument('--val_steps', type=int,
                        default=1, help='每隔多少步在验证集上评估一次')
    parser.add_argument('--lr', type=float, default=0.1, help='初始学习率')
    parser.add_argument('--path_to_company_model_save_dir',
                        type=str, required=False, help='Company端模型保存路径')
    parser.add_argument('--path_to_partner_model_save_dir',
                        type=str, required=False, help='Partner端模型保存路径')
    parser.add_argument('--reg_coef', type=float,
                        default=0.0, help='XGBoost正则化系数')
    parser.add_argument('--n_estimators', type=int,
                        default=2, help='XGBoost决策树数量')
    parser.add_argument('--max_depth', type=int,
                        default=2, help='XGBoost树最大深度')
    parser.add_argument('--K_quantiles', type=int,
                        default=1, help='用于联邦XGBoost的量化分位数')
    parser.add_argument('--path_to_buckets', type=str,
                        required=False, help='联邦XGBoost量化分位数保存路径')
    args = parser.parse_args()
    main(args)