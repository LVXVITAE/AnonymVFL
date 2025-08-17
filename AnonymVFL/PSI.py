from common import  out_dom
from heu import phe
from heu import numpy as hnp
import numpy as np
from hashlib import sha512
from rbcl import crypto_core_ristretto255_from_hash, crypto_core_ristretto255_scalar_random, crypto_scalarmult_ristretto255
import cloudpickle as pickle
import pandas as pd

from time import time
# 所有加密数据都按浮点数处理
encoder = phe.FloatEncoder(phe.SchemaType.ZPaillier)
class PSIWorker:
    """
    PSIWorker是PSICompany和PSIPartner的基类，提供了数据集读取和对本方持有数据进行加密的功能。
    """
    def __init__(self, keys : pd.DataFrame | list[str], private_features : pd.DataFrame | np.ndarray, public_features : pd.DataFrame | np.ndarray = None)-> None:
        """
        ## Args:
         - keys: 参与方的唯一标识符，要求是一个字符串列表。
         - private_features: 参与方持有的私有特征数据，即需要加密的特征。
         - public_features: 参与方持有的公开特征数据，默认为None。这个参数是预留给XGBoost分桶标签使用。
        """
        if isinstance(keys, pd.DataFrame):
            keys = keys.values.tolist()
        self.keys = keys
        if isinstance(private_features, pd.DataFrame):
            private_features = private_features.to_numpy()
        self.private_features = private_features.astype(np.float32)
        self.public_features = public_features
        if isinstance(public_features, pd.DataFrame):
            public_features = public_features.to_numpy()
            self.public_features = public_features.astype(int)

        
        self.kit = hnp.setup(phe.SchemaType.ZPaillier, 2048)
        self.private_features = self.kit.array(self.private_features, encoder=encoder)
        self.k = crypto_core_ristretto255_scalar_random()
        self.num_features = self.private_features.shape[1]
        self.num_records = self.private_features.shape[0]
    def hash_enc_raw(self):
        """
        对keys进行哈希处理，并将私有特征进行加密，最后key、私有特征和公开特征进行随机排列。
        ## Returns:
         - U_0: 哈希后的keys列表
         - U_1: 加密后的私有特征
         - U_2: 公开特征（如果存在）
        """
        U_0 = [crypto_core_ristretto255_from_hash(sha512(key.encode()).digest()) for key in self.keys]
        encryptor = self.kit.encryptor()
        U_1 = encryptor.encrypt(self.private_features)
        U_2 = self.public_features

        pem = np.random.permutation(self.num_records).tolist()
        # 对U_0, U_1, U_2进行随机排列
        U_0 = [crypto_scalarmult_ristretto255(self.k,U_0[i]) for i in pem] # 哈希值乘k次方
        U_1 = U_1[pem]
        if U_2 is not None:
            U_2 = U_2[pem]
        return U_0, U_1, U_2

class PSICompany(PSIWorker):
    def exchange(self):
        """
        ## Returns:
         - U_c: 包含乘方后的key哈希值、加密后的私有特征
         - pk_buffer: Company公钥
        """
        pk_buffer = pickle.dumps(self.kit.public_key())
        U_c = self.hash_enc_raw()
        return U_c, pk_buffer
    def compute_intersection(self,E_c ,U_p, partner_pk):
        """
        计算交集
        ## Args:
         - E_c: Company加密后的数据，包含二次乘方后的key哈希值、加密后的私有特征和公开特征
         - U_p: Partner加密后的数据，包含乘方后的keys哈希值、加密后的私有特征和公开特征
         - partner_pk: Partner的公钥
        ## Returns:
         - L: 交集的索引和未解密的Partner私有特征分片
         - R_cI: Company私有特征分片和公开特征。其中R_cI[0]是私有特征分片，R_cI[1]是公开特征。
        """
        U_p_0, U_p_1, U_p_2 = U_p
        peer_num_records, peer_num_features = U_p_1.shape
        # 计算Partner二次乘方后的哈希值
        E_p_0 = [crypto_scalarmult_ristretto255(self.k,u_p_0_i) for u_p_0_i in U_p_0]
        E_p_1, E_p_2 = U_p_1, U_p_2

        E_c_0, E_c_1, E_c_2 = E_c

        self.pkit = hnp.setup(pickle.loads(partner_pk))
        # 比较二次乘方后的哈希值求交集
        # 此处可考虑针对非平衡数据集场景进行优化
        company_hash = pd.DataFrame([(ec0, i) for i, ec0 in enumerate(E_c_0)],columns=['hash','i'])
        partner_hash = pd.DataFrame([(ep0, j) for j, ep0 in enumerate(E_p_0)],columns=['hash','j'])
        intersection = pd.merge(company_hash,partner_hash,how='inner',on='hash')
        # 生成随机数。理论上随机数的范围应是Paillier的明文空间，但实际上小一些的值也不影响结果的正确性
        # 后续可研究如何获取Paillier的明文空间的值
        r_p = np.random.randint(0, out_dom, size=(len(intersection), peer_num_features))
        r_p_enc = self.pkit.encryptor().encrypt(self.pkit.array(r_p, encoder=encoder))

        print("Computing masked partner cipher")
        L = (
            intersection['i'].values.tolist(), 
            # homo sub
            self.pkit.evaluator().sub(E_p_1[intersection['j'].tolist()], r_p_enc),
            E_p_2[intersection['j'].values] if E_p_2 is not None else None
        )    
        print("Computing company shares")  
        R_cI = (
            self.kit.decryptor().decrypt(E_c_1[intersection['i'].tolist()]).to_numpy(encoder),
            r_p
        )
        return L, (np.hstack((R_cI[0], R_cI[1])),E_c_2[intersection['i'].tolist()] if E_c_2 is not None else None)
            

class PSIPartner(PSIWorker):
    def exchange(self,U_c, company_pk):  
        """
        交换数据和公钥
        ## Args:
         - U_c: Company加密后的数据，包含乘方后的key哈希值、加密后的私有特征和公开特征
         - company_pk: Company的公钥
        ## Returns:
         - E_c: Company加密后的数据，包含二次乘方后的keys哈希值、未解密的私有特征分片和公开特征
         - U_p: Partner乘方keys哈希值、加密后的私有特征和公开特征
         - pk_buffer: Partner公钥
        """
        pk_buffer = pickle.dumps(self.kit.public_key())
        U_p = self.hash_enc_raw()

        self.kit_c = hnp.setup(pickle.loads(company_pk))

        U_c_0, U_c_1, U_c_2 = U_c
        peer_num_records, peer_num_features = U_c_1.shape
        # 生成随机数。理论上随机数的范围应是Paillier的明文空间，但实际上小一些的值也不影响结果的正确性
        # 后续可研究如何获取Paillier的明文空间的值
        self.r_c = np.random.randint(0, out_dom, size=U_c_1.shape)

        print("Computing masked company cipher")
        # 计算Company二次乘方后的哈希值
        E_c_0 = [crypto_scalarmult_ristretto255(self.k,u_c_0_i) for u_c_0_i in U_c_0] 
        r_c_enc = self.kit_c.encryptor().encrypt(self.kit_c.array(self.r_c, encoder=encoder))
        # homo sub
        E_c_1 = self.kit_c.evaluator().sub(U_c_1, r_c_enc)
        E_c_2 = U_c_2

        pem = np.random.permutation(peer_num_records).tolist()
        E_c_0 = [E_c_0[i] for i in pem]
        E_c_1 = E_c_1[pem]
        if E_c_2 is not None:
            E_c_2 = E_c_2[pem]

        self.r_c = self.r_c[pem]

        return (E_c_0, E_c_1, E_c_2), U_p, pk_buffer

    def output_shares(self, L):
        """
        输出Partner的私有特征分片和公开特征
        ## Args:
            L: 包含交集索引、未解密的Partner私有特征分片和公开特征
        ## Returns:
            R_pI: Partner的私有特征分片和公开特征。其中R_pI[0]是私有特征分片，R_pI[1]是公开特征。
        """

        print("Computing partner shares")
        r_c = self.r_c[L[0]]
        R_pI = (
            r_c,
            self.kit.decryptor().decrypt(L[1]).to_numpy(encoder)
        )
        return (np.hstack((R_pI[0],R_pI[1])), L[2])

# 下面的代码是生成随机数据集测试PSI性能，直接运行本文件即可
def generate_random_data(num_records, num_features):
    import random,string

    keys = [''.join(random.choices(string.ascii_uppercase +
                             string.digits, k=20)) for _ in range(num_records)]
    keys = pd.DataFrame(keys,columns=['0'])
    data = pd.DataFrame(100*np.random.rand(num_records,num_features))
    data = pd.concat([keys,data],axis=1)
    data.columns = range(data.shape[1])
    return data

def test_PSI(company_data, partner_data, INTER_ori):
    t1 = time()
    company = PSICompany(company_data.iloc[:,0], company_data.iloc[:,1:])
    partner = PSIPartner(partner_data.iloc[:,0], partner_data.iloc[:,1:])
    U_c, company_pk = company.exchange()
    E_c, U_p, partner_pk = partner.exchange(U_c, company_pk)
    L, R_cI = company.compute_intersection(E_c, U_p, partner_pk)
    R_pI = partner.output_shares(L)
    print("PSI time taken: ",time()-t1)
    R_I = (R_cI[0] + R_pI[0]) % out_dom
    print(R_I)
    
   # 获取原始交集数据（不包含ID列）
    intersection_records = pd.merge(company_data, partner_data, how='inner')
    intersection_features = intersection_records.iloc[:, 1:].to_numpy()
    
    # 复制特征以匹配R_I的形式（模拟合并后的数据）
    expected_data = np.concatenate([intersection_features, intersection_features], axis=1)
    
    # 检查记录数是否匹配
    print(f"预期交集记录数: {len(expected_data)}")
    print(f"PSI结果记录数: {len(R_I)}")
    
    if len(expected_data) != len(R_I):
        print(f"❌ 验证失败: 记录数不匹配! 预期 {len(expected_data)}, 实际得到 {len(R_I)}")
        return
    
    # 排序两个数组以便按行比较（如果行的顺序可能不同）
    # 注意：这里使用第一列作为排序依据，实际应用中可能需要更复杂的排序方式
    expected_data = expected_data[expected_data[:, 0].argsort()]
    R_I = R_I[R_I[:, 0].argsort()]
    
    # 使用numpy.isclose进行近似比较，允许万分之一的相对误差
    is_close = np.isclose(expected_data, R_I, rtol=1e-4)
    match_ratio = np.mean(is_close)
    
    # 检查所有元素是否都近似相等
    if np.all(is_close):
        print(f"✅ 验证成功: 所有值都在允许误差范围内匹配! (匹配率: 100%)")
    else:
        print(f"⚠️ 部分验证: 部分值超出允许误差范围 (匹配率: {match_ratio:.2%})")
        # 找出不匹配的位置
        mismatch_indices = np.where(~is_close)
        mismatch_count = len(mismatch_indices[0])
        print(f"   - 不匹配元素数: {mismatch_count} / {expected_data.size}")
        
        # 显示部分不匹配的样例（最多5个）
        if mismatch_count > 0:
            print("   - 不匹配样例 (预期值 vs 实际值):")
            for i in range(min(5, mismatch_count)):
                row, col = mismatch_indices[0][i], mismatch_indices[1][i]
                print(f"     [{row},{col}]: {expected_data[row,col]} vs {R_I[row,col]}, 差值: {abs(expected_data[row,col] - R_I[row,col])}")

def test1_PSI(company_data, partner_data, INTER_ori):
    """
    测试PSI（Private Set Intersection）功能
    
    Args:
        company_data: 公司方数据 - DataFrame格式，第0列为键，其余列为特征
        partner_data: 合作方数据 - DataFrame格式，第0列为键，其余列为特征
        INTER_ori: 原始交集数据（用于验证）
    """
    print("=" * 60)
    print("开始PSI测试流程")
    print("=" * 60)
    
    # ========== 步骤1: 输入数据分析 ==========
    print("\n【步骤1: 输入数据分析】")
    print(f"Company数据形状: {company_data.shape}")
    print(f"Partner数据形状: {partner_data.shape}")
    print(f"Company数据列名: {company_data.columns.tolist()}")
    print(f"Partner数据列名: {partner_data.columns.tolist()}")
    
    # 显示前几行数据示例
    print(f"\nCompany数据前3行:")
    print(company_data.head(3))
    print(f"\nPartner数据前3行:")
    print(partner_data.head(3))
    
    # 分析键的重叠情况
    company_keys = set(company_data.iloc[:, 0])
    partner_keys = set(partner_data.iloc[:, 0])
    expected_intersection = company_keys & partner_keys
    print(f"\n预期交集大小（基于键匹配）: {len(expected_intersection)}")
    
    # ========== 步骤2: 初始化PSI对象 ==========
    print("\n【步骤2: 初始化PSI对象】")
    t1 = time()
    
    # 创建PSI对象，分离键和特征
    company = PSICompany(company_data.iloc[:,0], company_data.iloc[:,1:])
    partner = PSIPartner(partner_data.iloc[:,0], partner_data.iloc[:,1:])
    
    print(f"Company PSI对象 - 记录数: {company.num_records}, 特征数: {company.num_features}")
    print(f"Partner PSI对象 - 记录数: {partner.num_records}, 特征数: {partner.num_features}")
    print(f"Company公私钥对已生成，密钥长度: 2048位")
    print(f"Partner公私钥对已生成，密钥长度: 2048位")
    
    # ========== 步骤3: Company方数据交换 ==========
    print("\n【步骤3: Company方数据准备和交换】")
    U_c, company_pk = company.exchange()
    
    print(f"Company exchange()返回:")
    print(f"  - U_c类型: {type(U_c)}, 长度: {len(U_c)}")
    print(f"  - U_c[0] (哈希后的键): {len(U_c[0])}个哈希值")
    print(f"  - U_c[1] (加密特征): 形状 {U_c[1].shape}")
    print(f"  - U_c[2] (公开特征): {U_c[2] if U_c[2] is not None else 'None'}")
    print(f"  - 公钥大小: {len(company_pk)}字节")
    
    # 显示哈希值示例（前3个）
    print(f"  - 哈希值示例（前3个）: {[hex(int.from_bytes(h[:8], 'big')) for h in U_c[0][:3]]}")
    
    # ========== 步骤4: Partner方数据交换 ==========
    print("\n【步骤4: Partner方数据准备和交换】")
    E_c, U_p, partner_pk = partner.exchange(U_c, company_pk)
    
    print(f"Partner exchange()返回:")
    print(f"  - E_c类型: {type(E_c)}, 长度: {len(E_c)}")
    print(f"  - E_c[0] (Company二次哈希): {len(E_c[0])}个哈希值")
    print(f"  - E_c[1] (Company掩码特征): 形状 {E_c[1].shape}")
    print(f"  - E_c[2] (Company公开特征): {E_c[2] if E_c[2] is not None else 'None'}")
    print(f"  - U_p[0] (Partner哈希键): {len(U_p[0])}个哈希值")
    print(f"  - U_p[1] (Partner加密特征): 形状 {U_p[1].shape}")
    print(f"  - U_p[2] (Partner公开特征): {U_p[2] if U_p[2] is not None else 'None'}")
    print(f"  - Partner公钥大小: {len(partner_pk)}字节")
    
    # ========== 步骤5: 计算交集 ==========
    print("\n【步骤5: Company方计算交集】")
    L, R_cI = company.compute_intersection(E_c, U_p, partner_pk)
    
    print(f"Company compute_intersection()返回:")
    print(f"  - L类型: {type(L)}, 长度: {len(L)}")
    print(f"  - L[0] (交集索引): {len(L[0])}个索引")
    print(f"    索引示例: {L[0][:5] if len(L[0]) >= 5 else L[0]}")
    print(f"  - L[1] (Partner掩码特征密文): 形状 {L[1].shape}")
    print(f"  - L[2] (Partner公开特征): {L[2] if L[2] is not None else 'None'}")
    
    print(f"  - R_cI类型: {type(R_cI)}, 长度: {len(R_cI)}")
    print(f"  - R_cI[0] (合并后Company分片): 形状 {R_cI[0].shape}")
    print(f"    数据类型: {R_cI[0].dtype}, 数据范围: [{R_cI[0].min():.2f}, {R_cI[0].max():.2f}]")
    print(f"  - R_cI[1] (Company公开特征): {R_cI[1] if R_cI[1] is not None else 'None'}")
    
    # ========== 步骤6: Partner输出分片 ==========
    print("\n【步骤6: Partner方输出分片】")
    R_pI = partner.output_shares(L)
    
    print(f"Partner output_shares()返回:")
    print(f"  - R_pI类型: {type(R_pI)}, 长度: {len(R_pI)}")
    print(f"  - R_pI[0] (合并后Partner分片): 形状 {R_pI[0].shape}")
    print(f"    数据类型: {R_pI[0].dtype}, 数据范围: [{R_pI[0].min():.2f}, {R_pI[0].max():.2f}]")
    print(f"  - R_pI[1] (Partner公开特征): {R_pI[1] if R_pI[1] is not None else 'None'}")
    
    psi_time = time() - t1
    print(f"\n总PSI计算时间: {psi_time:.4f}秒")
    
    # ========== 步骤7: 恢复完整数据 ==========
    print("\n【步骤7: 恢复完整数据（秘密分享重构）】")
    print("执行加法重构: R_I = (R_cI[0] + R_pI[0]) % out_dom")
    R_I = (R_cI[0] + R_pI[0]) % out_dom
    
    print(f"恢复的完整数据R_I:")
    print(f"  - 形状: {R_I.shape}")
    print(f"  - 数据类型: {R_I.dtype}")
    print(f"  - 数据范围: [{R_I.min():.2f}, {R_I.max():.2f}]")
    print(f"  - 前3行数据:")
    for i in range(min(3, len(R_I))):
        print(f"    行{i}: {R_I[i][:5]}..." if R_I.shape[1] > 5 else f"    行{i}: {R_I[i]}")
    
    # ========== 步骤8: 生成预期结果进行验证 ==========
    print("\n【步骤8: 验证PSI结果】")
    
    # 获取原始交集数据（不包含ID列）
    intersection_records = pd.merge(company_data, partner_data, how='inner')
    print(f"Pandas merge得到的原始交集记录数: {len(intersection_records)}")
    print(f"原始交集数据形状: {intersection_records.shape}")
    
    intersection_features = intersection_records.iloc[:, 1:].to_numpy()
    print(f"提取的交集特征形状: {intersection_features.shape}")
    
    # 复制特征以匹配R_I的形式（模拟PSI中的特征合并）
    # 注意：这里复制是因为测试中Company和Partner使用了相同的特征
    expected_data = np.concatenate([intersection_features, intersection_features], axis=1)
    print(f"构造的预期数据形状: {expected_data.shape}")
    print(f"预期数据前3行:")
    for i in range(min(3, len(expected_data))):
        print(f"    行{i}: {expected_data[i][:5]}..." if expected_data.shape[1] > 5 else f"    行{i}: {expected_data[i]}")
    
    # ========== 步骤9: 数据比较和验证 ==========
    print("\n【步骤9: 数据比较和验证】")
    
    # 检查记录数是否匹配
    print(f"记录数比较: 预期 {len(expected_data)} vs PSI结果 {len(R_I)}")
    
    if len(expected_data) != len(R_I):
        print(f"❌ 验证失败: 记录数不匹配!")
        return
    
    if len(expected_data) == 0:
        print(f"⚠️ 警告: 交集为空")
        return
    
    # 排序两个数组以便按行比较（PSI可能改变行的顺序）
    print("对数据进行排序以便比较...")
    expected_data_sorted = expected_data[expected_data[:, 0].argsort()]
    R_I_sorted = R_I[R_I[:, 0].argsort()]
    
    # 使用numpy.isclose进行近似比较，允许浮点数误差
    print("执行数值比较（允许1e-4的相对误差）...")
    is_close = np.isclose(expected_data_sorted, R_I_sorted, rtol=1e-4)
    match_ratio = np.mean(is_close)
    total_elements = expected_data_sorted.size
    matched_elements = np.sum(is_close)
    
    print(f"\n验证结果:")
    print(f"  - 总元素数: {total_elements}")
    print(f"  - 匹配元素数: {matched_elements}")
    print(f"  - 匹配率: {match_ratio:.4f} ({match_ratio*100:.2f}%)")
    
    # 检查所有元素是否都近似相等
    if np.all(is_close):
        print(f"✅ 验证成功: 所有值都在允许误差范围内匹配!")
    else:
        print(f"⚠️ 部分验证: 部分值超出允许误差范围")
        # 找出不匹配的位置
        mismatch_indices = np.where(~is_close)
        mismatch_count = len(mismatch_indices[0])
        print(f"   - 不匹配元素数: {mismatch_count}")
        
        # 显示部分不匹配的样例（最多5个）
        if mismatch_count > 0:
            print("   - 不匹配样例 (预期值 vs 实际值):")
            for i in range(min(5, mismatch_count)):
                row, col = mismatch_indices[0][i], mismatch_indices[1][i]
                expected_val = expected_data_sorted[row, col]
                actual_val = R_I_sorted[row, col]
                diff = abs(expected_val - actual_val)
                print(f"     [{row},{col}]: {expected_val:.6f} vs {actual_val:.6f}, 差值: {diff:.6f}")
    
    print("\n" + "=" * 60)
    print("PSI测试流程完成")
    print("=" * 60)

def random_PSI_test(scale = 100):
    intersection = generate_random_data(10,100)
    company_data = pd.concat([generate_random_data(scale,100),intersection],axis=0).sample(frac=1)
    partner_data = pd.concat([generate_random_data(scale//2,100),intersection],axis=0).sample(frac=1)
    intersection = pd.merge(company_data,partner_data,how='inner')
    intersection = intersection.to_numpy()[:,1:]
    intersection = np.append(intersection,intersection,axis=1)
    print(intersection)
    INTER_ori = set()
    for I in intersection:
        H_I = sha512()
        for i in I:
            H_I.update(str(i).encode())
        INTER_ori.add(H_I.hexdigest())
    test1_PSI(company_data, partner_data, INTER_ori)

if __name__ == "__main__":
    random_PSI_test()