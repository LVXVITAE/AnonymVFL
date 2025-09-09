import pandas as pd
import numpy as np
import os

def process_online_shoppers_data():
    """
    处理online_shoppers_intention.csv文件，进行数据预处理并拆分成训练和测试集
    """
    
    # 1. 读取原始数据
    input_file = '/home/dxn/mobile_project2/AnonymVFL/Datasets/data/data/online_shoppers_intention.csv'
    df = pd.read_csv(input_file)
    
    print(f"原始数据形状: {df.shape}")
    print(f"列名: {df.columns.tolist()}")
    
    # 2. 数据类型转换
    # 将Month转换为数值型 (1-12)
    month_mapping = {
        'Jan': 1, 'Feb': 2, 'Mar': 3, 'May': 5, 'June': 6, 'Jul': 7,
        'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
    }
    df['Month'] = df['Month'].map(month_mapping)
    
    # 将VisitorType转换为数值型 (1-3)
    visitor_type_mapping = {
        'Returning_Visitor': 1,
        'New_Visitor': 2,
        'Other': 3
    }
    df['VisitorType'] = df['VisitorType'].map(visitor_type_mapping)
    
    # 将Weekend和Revenue的True/False转换为1/0
    df['Weekend'] = df['Weekend'].astype(int)
    df['Revenue'] = df['Revenue'].astype(int)
    
    print("数据类型转换完成")
    
    # 3. 添加ID列（从0开始）
    df.insert(0, 'id', range(len(df)))
    
    print(f"添加ID列后的数据形状: {df.shape}")
    print("处理后的前5行数据:")
    print(df.head())
    
    # 4. 定义列索引
    # 原始18列 + 新增的id列 = 19列
    # Host方: id列 + 第1-8列 + 最后1列 (id, 0-7, 18) -> (0, 1-8, 18)
    # Guest方: id列 + 第9-17列 (id, 8-16) -> (0, 9-17)
    
    host_cols = [0] + list(range(1, 9)) + [18]  # id + 前8列 + 最后1列
    guest_cols = [0] + list(range(9, 18))       # id + 第9-17列
    
    print(f"Host列索引: {host_cols}")
    print(f"Guest列索引: {guest_cols}")
    print(f"Host列名: {[df.columns[i] for i in host_cols]}")
    print(f"Guest列名: {[df.columns[i] for i in guest_cols]}")
    
    # 5. 创建输出目录
    output_dir = 'Datasets/processed'
    os.makedirs(output_dir, exist_ok=True)
    
    # # 6. 拆分数据集
    
    # # Host训练集: id 0-8999 (9000条数据)
    # host_train = df.loc[df['id'].between(0, 8999), df.columns[host_cols]].copy()
    
    # # Guest训练集: id 1000-9999 (9000条数据)
    # guest_train = df.loc[df['id'].between(1000, 9999), df.columns[guest_cols]].copy()
    
    # # Host测试集: id 10000及之后
    # host_test = df.loc[df['id'] >= 10000, df.columns[host_cols]].copy()
    
    # # Guest测试集: id 10000及之后
    # guest_test = df.loc[df['id'] >= 10000, df.columns[guest_cols]].copy()
    
    # 6. 拆分数据集
    # 随机选取10000个ID作为训练集
    train_ids = np.random.choice(df['id'].unique(), size=10000, replace=False)
    train_ids_set = set(train_ids)

    # 剩余的ID作为测试集
    test_ids = [id for id in df['id'].unique() if id not in train_ids_set]

    # Host训练集: 从10000个训练ID中随机选9000个
    host_train_ids = np.random.choice(train_ids, size=9000, replace=False)
    host_train = df.loc[df['id'].isin(host_train_ids), df.columns[host_cols]].copy()

    # Guest训练集: 从10000个训练ID中随机选9000个
    guest_train_ids = np.random.choice(train_ids, size=9000, replace=False)
    guest_train = df.loc[df['id'].isin(guest_train_ids), df.columns[guest_cols]].copy()

    # Host测试集: 使用剩余的ID
    host_test = df.loc[df['id'].isin(test_ids), df.columns[host_cols]].copy()

    # Guest测试集: 使用剩余的ID
    guest_test = df.loc[df['id'].isin(test_ids), df.columns[guest_cols]].copy()
    
    # 7. 保存文件
    host_train.to_csv(f'{output_dir}/host_train.csv', index=False)
    host_test.to_csv(f'{output_dir}/host_test.csv', index=False)
    guest_train.to_csv(f'{output_dir}/guest_train.csv', index=False)
    guest_test.to_csv(f'{output_dir}/guest_test.csv', index=False)
    
    # 8. 验证结果
    print("\n=== 数据拆分结果 ===")
    print(f"Host训练集形状: {host_train.shape} (期望: (9000, 10))")
    print(f"Guest训练集形状: {guest_train.shape} (期望: (9000, 10))")
    print(f"Host测试集形状: {host_test.shape} (期望: (2330, 10))")
    print(f"Guest测试集形状: {guest_test.shape} (期望: (2330, 10))")
    
    print(f"\nHost训练集ID范围: {host_train['id'].min()} - {host_train['id'].max()}")
    print(f"Guest训练集ID范围: {guest_train['id'].min()} - {guest_train['id'].max()}")
    print(f"Host测试集ID范围: {host_test['id'].min()} - {host_test['id'].max()}")
    print(f"Guest测试集ID范围: {guest_test['id'].min()} - {guest_test['id'].max()}")
    
    # 9. 显示示例数据
    print("\n=== Host训练集前5行 ===")
    print(host_train.head())
    
    print("\n=== Guest训练集前5行 ===")
    print(guest_train.head())
    
    print("\n=== Host测试集前5行 ===")
    print(host_test.head())
    
    print("\n=== Guest测试集前5行 ===")
    print(guest_test.head())
    
    print(f"\n所有文件已保存到 {output_dir}/ 目录下")
    
    return host_train, guest_train, host_test, guest_test

if __name__ == "__main__":
    process_online_shoppers_data()
