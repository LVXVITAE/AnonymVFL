import os
import numpy as np
from common import *

from SharedVariable import SharedVariable
from LR import train, SSLR_test

def mul_test():
    # X = np.random.randint(0, out_dom, (10,10), np.int64)
    X = 0.001*np.random.rand(10,10000)
    # X = np.zeros((10,10),dtype=np.float64)
    Y = np.random.randint(0, 1000, (10000,10), dtype=np.int64)
    Y.astype(np.float64)
    XY = X @ Y
    # XY = X * Y
    X = SharedVariable.from_secret(X)
    Y = SharedVariable.from_secret(Y)
    # Z = (X * Y).reveal()
    # assert np.allclose(Z, XY), "Secure multiplication failed!"
    Z = (X @ Y).reveal()
    assert np.allclose(Z, XY), "Secure matmul failed!"

def LT_test():
    X = np.random.randint(0,4,(10000,1))
    Y = np.ones_like(X)
    LT = X < Y
    LT = LT.reshape(-1)
    X0 = SharedVariable.from_secret(X)
    Y0 = SharedVariable.from_secret(Y)
    Z = X0 < Y0
    Z = Z.reshape(-1)
    if not (Z == LT).all():
        i = np.argwhere(Z != LT).reshape(-1)
        print(X[i].T)


def PSI_LR():
    # 导入所需库
    from sklearn.model_selection import train_test_split
    from random import choices
    import string    
    import pandas as pd
    from PSI import PSICompany, PSIPartner
    from LR import train, SSLR_test
    
    # ========== 步骤1: 数据预处理 ==========
    print("=" * 60)
    print("PSI_LR 流程开始")
    print("=" * 60)
    
    data = pd.read_csv(os.path.join("Datasets","pcs.csv"),delimiter=',').astype(np.uint64)
    print(f"原始数据形状: {data.shape}")
    print(f"原始数据列数: {data.columns.tolist()}")
    
    # 标准划分：最后一列是标签
    original_features = data.iloc[:, :-1]  # 前n-1列是特征
    original_labels = data.iloc[:, -1]     # 最后一列是标签
    print(f"原始特征形状: {original_features.shape}")
    print(f"原始标签形状: {original_labels.shape}")
    
    # 分割训练测试集
    train_data, test_data = train_test_split(data, test_size=0.25, random_state=42)
    print(f"训练数据形状: {train_data.shape}")
    print(f"测试数据形状: {test_data.shape}")
    
    # ========== 步骤2: 准备PSI数据 ==========
    print(f"\n【步骤2: 准备PSI数据】")
    
    # 重置索引
    train_data.index = range(train_data.shape[0])
    
    # 为训练数据添加唯一密钥
    keys = [''.join(choices(string.ascii_uppercase + string.digits, k=20)) 
            for _ in range(train_data.shape[0])]
    keys_df = pd.DataFrame(keys, columns=['key'])
    
    # 构建包含密钥的训练数据：[密钥, 特征, 标签]
    train_data_with_keys = pd.concat([keys_df, train_data], axis=1)
    print(f"添加密钥后训练数据形状: {train_data_with_keys.shape}")
    print(f"数据列结构: ['key'] + {train_data.columns.tolist()}")
    
    # ========== 步骤3: 垂直分割数据 ==========
    print(f"\n【步骤3: 垂直分割数据模拟】")
    
    # 计算特征分割点（假设特征平均分割）
    n_features = train_data.shape[1] - 1  # 减去标签列
    split_point = n_features // 2
    
    # Company方：密钥 + 前半部分特征 + 标签
    # Partner方：密钥 + 后半部分特征
    
    # 随机采样模拟现实中的数据不完全重叠
    company_sample = train_data_with_keys.sample(frac=0.9, random_state=42)
    partner_sample = train_data_with_keys.sample(frac=0.9, random_state=43)
    
    # Company方数据：[密钥, 前split_point个特征, 标签]
    company_feature_cols = ['key'] + train_data.columns[:split_point].tolist() + [train_data.columns[-1]]
    company_data = company_sample[company_feature_cols]
    
    # Partner方数据：[密钥, 后面的特征]  
    partner_feature_cols = ['key'] + train_data.columns[split_point:-1].tolist()
    partner_data = partner_sample[partner_feature_cols]
    
    print(f"Company方数据形状: {company_data.shape}")
    print(f"Company方列: {company_data.columns.tolist()}")
    print(f"Partner方数据形状: {partner_data.shape}")
    print(f"Partner方列: {partner_data.columns.tolist()}")
    
    # ========== 步骤4: 执行PSI ==========
    print(f"\n【步骤4: 执行PSI】")
    
    # PSI输入：第0列是密钥，其余列是数据（特征+标签）
    company_psi = PSICompany(company_data.iloc[:,0], company_data.iloc[:,1:])
    partner_psi = PSIPartner(partner_data.iloc[:,0], partner_data.iloc[:,1:])
    
    print(f"Company PSI - 记录数: {company_psi.num_records}, 数据列数: {company_psi.num_features}")
    print(f"Partner PSI - 记录数: {partner_psi.num_records}, 数据列数: {partner_psi.num_features}")
    
    U_c, company_pk = company_psi.exchange()
    E_c, U_p, partner_pk = partner_psi.exchange(U_c, company_pk)
    L, R_cI = company_psi.compute_intersection(E_c, U_p, partner_pk)
    R_pI = partner_psi.output_shares(L)
    
    intersection_size = len(L[0]) if L and hasattr(L[0], '__len__') else 0
    print(f"PSI完成，交集大小: {intersection_size}")
    
    if intersection_size == 0:
        print("❌ 错误: PSI交集为空，无法继续训练")
        return None
    
    # ========== 步骤5: 重构PSI后的数据 ==========
    print(f"\n【步骤5: 重构PSI后的数据】")
    
    # 恢复完整数据
    R_cI_data = np.array(R_cI[0], dtype=np.int64) - out_dom
    R_pI_data = np.array(R_pI[0], dtype=np.int64)
    
    print(f"Company分片形状: {R_cI_data.shape}")
    print(f"Partner分片形状: {R_pI_data.shape}")
    
    # 重构完整数据：R_I = (R_cI + R_pI) % out_dom
    R_I = (R_cI_data + R_pI_data) % out_dom
    print(f"重构后完整数据形状: {R_I.shape}")
    
    # ========== 步骤6: 分离特征和标签 ==========
    print(f"\n【步骤6: 分离特征和标签】")
    
    # 分析重构数据的结构
    # 根据PSI的实现，R_I包含了两方的数据合并结果
    # 需要根据原始的数据分割方式来正确提取特征和标签
    
    # Company方原始数据结构: [前split_point个特征, 标签]
    # Partner方原始数据结构: [后面的特征]
    # 重构后的数据应该是: [Company特征, Company标签, Partner特征, Partner标签(0填充)]
    
    company_features_end = split_point
    company_label_idx = company_features_end
    partner_features_start = company_label_idx + 1
    partner_features_end = partner_features_start + (n_features - split_point)
    
    # 提取Company的特征和标签
    train_X_company = R_I[:, :company_features_end]  # Company特征
    train_y_psi = R_I[:, company_label_idx:company_label_idx+1]  # Company标签
    train_X_partner = R_I[:, partner_features_start:partner_features_end]  # Partner特征
    
    # 合并所有特征
    train_X_psi = np.concatenate([train_X_company, train_X_partner], axis=1)
    
    print(f"PSI后训练特征形状: {train_X_psi.shape}")
    print(f"PSI后训练标签形状: {train_y_psi.shape}")
    print(f"特征范围: [{train_X_psi.min():.2f}, {train_X_psi.max():.2f}]")
    print(f"标签唯一值: {np.unique(train_y_psi)}")
    
    # ========== 步骤7: 准备测试数据 ==========
    print(f"\n【步骤7: 准备测试数据】")
    
    # 测试数据保持原始格式：最后一列是标签
    test_X = test_data.iloc[:, :-1].values  # 所有特征
    test_y = test_data.iloc[:, -1].values.reshape(-1, 1)  # 标签
    
    print(f"测试特征形状: {test_X.shape}")
    print(f"测试标签形状: {test_y.shape}")
    
    # 检查特征维度匹配
    if train_X_psi.shape[1] != test_X.shape[1]:
        print(f"⚠️  特征维度不匹配: 训练{train_X_psi.shape[1]} vs 测试{test_X.shape[1]}")
        
        if train_X_psi.shape[1] > test_X.shape[1]:
            # 训练特征多，截取前面的特征
            train_X_psi = train_X_psi[:, :test_X.shape[1]]
            print(f"截取训练特征到: {train_X_psi.shape}")
        else:
            # 测试特征多，截取前面的特征
            test_X = test_X[:, :train_X_psi.shape[1]]
            print(f"截取测试特征到: {test_X.shape}")
    
    # ========== 步骤8: 使用SharedVariable包装训练数据 ==========
    print(f"\n【步骤8: 准备安全计算数据】")
    
    # 为安全计算创建SharedVariable
    # 这里假设标签只在Company方，Partner方标签为0
    train_X_shared = SharedVariable(train_X_psi[:len(train_X_psi)//2], 
                                   train_X_psi[len(train_X_psi)//2:])
    train_y_shared = SharedVariable(train_y_psi[:len(train_y_psi)//2], 
                                   np.zeros_like(train_y_psi[len(train_y_psi)//2:]))
    
    print(f"SharedVariable训练特征形状: {train_X_shared.shape()}")
    print(f"SharedVariable训练标签形状: {train_y_shared.shape()}")
    
    # ========== 步骤9: 输出最终结果 ==========
    print(f"\n【步骤9: 数据准备完成】")
    print("=" * 60)
    print("PSI后的数据已准备就绪:")
    print(f"- train_X: {train_X_shared.shape()} (SharedVariable)")
    print(f"- train_y: {train_y_shared.shape()} (SharedVariable)")  
    print(f"- test_X: {test_X.shape} (numpy array)")
    print(f"- test_y: {test_y.shape} (numpy array)")
    print("=" * 60)
    
    # ========== 步骤10: 训练模型 ==========
    # print(f"\n【步骤10: 训练PSI-LR模型】")
    # weight = train(train_X_shared, train_y_shared, test_X, test_y).reveal()
    # print(f"模型训练完成，权重形状: {weight.shape}")
    
    print("train_X_shared的形状:", train_X_shared.shape)
    print("train_y_shared的形状:", train_y_shared.shape)
    print("test_X的形状:", test_X.shape)
    print("test_y的形状:", test_y.shape)
    
    train(train_X_shared, train_y_shared, test_X, test_y)

def sf_test():
    import secretflow as sf
    import jax.numpy as jnp

    # 初始化SecretFlow
    sf.init(['company', 'partner', 'coordinator'], address='local')
    # 获取集群定义
    aby3_config = sf.utils.testing.cluster_def(parties=['company', 'partner', 'coordinator'])
    # 创建SPU设备
    spu_device = sf.SPU(aby3_config)
    # 创建PYU设备
    company, partner, coordinator = sf.PYU('company'), sf.PYU('partner'), sf.PYU('coordinator')
    # 生成随机矩阵X
    X = 0.001*np.random.rand(10,10)
    # X = np.zeros((10,10),dtype=np.float64)
    # 生成随机矩阵Y
    Y = np.random.randint(0, 1000, (10,10), dtype=np.int64)
    # 将Y转换为浮点数
    Y.astype(np.float64)
    # 计算矩阵乘积XY
    XY = X @ Y

    # 将X、Y、XY转换为Jax数组
    X = jnp.array(X)
    Y = jnp.array(Y)
    XY = jnp.array(XY)
    # 计算X是否小于Y
    LT = X < Y
    # 打印矩阵乘积XY
    print(XY)
    # 打印X是否小于Y
    print(LT)

    # 将X、Y转换为PYU设备上的数据
    X = sf.to(company, X).to(spu_device)
    Y = sf.to(partner, Y).to(spu_device)
    # 定义矩阵乘法函数
    def matmul(x, y):
        return x @ y
    # 定义小于函数
    def LT(x, y):
        return x < y
    # 在SPU设备上执行矩阵乘法，并将结果转换为PYU设备上的数据
    # Z = spu_device(matmul)(X, Y).to(coordinator)
    Z = spu_device(LT)(X, Y).to(coordinator)
    # 将结果从PYU设备上取回
    Z = sf.reveal(Z)
    # 打印结果
    print(Z)


if __name__ == "__main__":
    # random_PSI_test()
    # LR_test()
    # random_mul_test()
    # LR_test()
    # SVM_test()
    PSI_LR()
    # sf_test()
    # PSI_SSLR_test('pcs')