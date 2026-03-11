from common import  out_dom
import numpy as np
from hashlib import sha512
# 导入Ristretto255椭圆曲线加密相关函数
from rbcl import crypto_core_ristretto255_from_hash, crypto_core_ristretto255_scalar_random, crypto_scalarmult_ristretto255
import pandas as pd
from time import time
# 导入SecretFlow框架
import secretflow as sf
from secretflow.device import PYUObject, HEUObject
from secretflow import HEU, PYU

# PSIWorker类：私有集合交集工作器的基类
class PSIWorker:
    """
    PSIWorker是PSICompany和PSIPartner的基类，提供了数据集读取和对本方持有数据进行加密的功能。
    该类实现了PS3I协议的基础功能。
    """
    def __init__(self, data : PYUObject, pyu_devices: tuple[PYU, PYU], heu_devices : tuple[HEU,HEU])-> None:
        """
        初始化PSI工作器，解包数据并生成随机私钥。
        ## Args:
         - data: PYUObject对象，包含(keys, private_features, public_features)三元组
         - pyu_devices: 两个PYU设备的元组，分别代表Company和Partner
         - heu_devices: 两个HEU设备的元组，分别用于Company和Partner的同态加密
        ## 内部属性:
         - keys: 参与方的唯一标识符，要求是一个字符串列表。
         - private_features: 参与方持有的私有特征数据，即需要加密的特征。
         - public_features: 参与方持有的公开特征数据，默认为None。这个参数是预留给XGBoost分桶标签使用。
         - k: 随机生成的私钥标量，用于DH密钥交换
        """
        # 获取数据所在的PYU设备
        self.device = data.device
        # 定义数据解包函数，将三元组拆分为keys、私有特征和公开特征
        def unpack(data : tuple[list[str], np.ndarray, np.ndarray | None]):
            keys, private_features, public_features = data
            return keys, private_features, public_features
        # 在PYU设备上执行解包操作，获取keys、私有特征和公开特征
        self.keys, self.private_features, self.public_features = self.device(unpack, num_returns=3)(data)
        # 生成随机私钥标量k，用于后续的DH密钥交换
        self.k = self.device(crypto_core_ristretto255_scalar_random)()
        # 获取私有特征的数据形状（行数和列数）
        self.data_shape = sf.reveal(self.device(np.shape)(self.private_features))
        # 存储双方的PYU设备和HEU设备，方便后续数据交换和加密操作
        self.company_heu, self.partner_heu = heu_devices
        self.company, self.partner = pyu_devices
        
    def hash_raw(self) -> tuple[PYUObject, HEUObject,PYUObject]:
        """
        对keys进行哈希处理，并将私有特征进行加密，最后key、私有特征和公开特征进行随机排列。
        这是PS3I协议的第一步：对ID进行哈希和标量乘法。
        ## Returns:
         - U_0: 哈希并乘以私钥后的keys列表
         - U_1: 私有特征
         - U_2: 公开特征（如果存在）
        """
        # 初始化三元组：keys、私有特征、公开特征
        U_0, U_1, U_2 = self.keys, self.private_features, self.public_features
        # 定义随机重排函数，打乱数据顺序以保护隐私
        def repermute(U_0, U_1, U_2):
            # 按随机索引重排keys
            pem = np.random.permutation(len(U_0)).tolist()
            U_0 = [U_0[i] for i in pem]
            # 按随机索引重排私有特征
            U_1 = U_1[pem]
            # 如果存在公开特征，也按随机索引重排
            if U_2 is not None:
                U_2 = U_2[pem]
            return U_0, U_1, U_2
        # 在PYU设备上执行随机重排
        U_0, U_1, U_2 = self.device(repermute,num_returns=3)(U_0, U_1, U_2)

        # 定义哈希和标量乘法函数
        def hash_mul_keys(keys, k):
            # 对每个key进行SHA512哈希，然后映射到Ristretto255椭圆曲线上的点
            keys = [crypto_core_ristretto255_from_hash(sha512(key.encode()).digest()) for key in keys]
            # 对每个椭圆曲线点乘以私钥k，实现DH密钥交换的第一步
            return [crypto_scalarmult_ristretto255(k, key) for key in keys]
        # 在PYU设备上执行哈希和标量乘法操作
        U_0 = self.device(hash_mul_keys)(U_0, self.k)
        # 返回处理后的三元组
        return (U_0, U_1, U_2)

# PSICompany类：Company方的PSI实现
class PSICompany(PSIWorker):
    """
    PSICompany继承自PSIWorker，代表Company方（主动方）的PSI实现。
    Company方负责发起交换请求并计算最终的交集结果。
    """
    def exchange(self):
        """
        Company方执行第一轮数据交换。
        将哈希后的keys发送给Partner，同时将私有特征用同态加密保护。
        ## Returns:
         - U_c: 包含乘方后的key哈希值、加密后的私有特征、公开特征的元组
         - data_shape: Company私有特征的数据形状
        """
        # 调用父类方法获取哈希处理后的数据
        U_c_0, U_c_1, U_c_2 = self.hash_raw()
        # 将哈希后的keys发送到Partner设备
        U_c_0 = U_c_0.to(self.partner)
        # 将私有特征发送到Company的HEU设备并进行同态加密
        U_c_1 = U_c_1.to(self.company_heu).encrypt()
        # 将公开特征发送到Partner设备
        U_c_2 = U_c_2.to(self.partner)
        # 返回处理后的数据元组和数据形状
        return (U_c_0, U_c_1, U_c_2), self.data_shape
    
    def compute_intersection(self,E_c ,U_p, partner_data_shape : tuple[int, int]):
        """
        计算交集并生成秘密共享分片。
        这是PS3I协议的核心步骤：比较双方二次乘方后的哈希值来找到交集。
        ## Args:
         - E_c: Company加密后的数据，包含二次乘方后的key哈希值、加密后的私有特征和公开特征
         - U_p: Partner加密后的数据，包含乘方后的keys哈希值、加密后的私有特征和公开特征
         - partner_data_shape: Partner私有特征的数据形状
        ## Returns:
         - L: 交集的索引和未解密的Partner私有特征分片（掩码后）
         - R_cI: Company私有特征分片和公开特征。其中R_cI[0]是私有特征分片，R_cI[1]是公开特征。
         - bucket_labels: 分桶标签（如有），即Company和Partner的公开特征。
        """
        # 解包Partner的数据
        U_p_0, U_p_1, U_p_2 = U_p

        # 定义标量乘法函数，用于计算Partner数据的二次乘方
        def mul_k(U_p_0, k):
            return [crypto_scalarmult_ristretto255(k,u_p_0_i) for u_p_0_i in U_p_0]
        # 对Partner的哈希值乘以Company的私钥，得到二次乘方后的哈希值
        E_p_0 = self.device(mul_k)(U_p_0, self.k)
        # E_p_1和E_p_2保持不变
        E_p_1, E_p_2 = U_p_1, U_p_2

        # 解包Company的数据
        E_c_0, E_c_1, E_c_2 = E_c

        # 定义交集索引计算函数
        def intersection_indices(E_c_0, E_p_0):
            '''比较二次乘方后的哈希值求交集
            通过比较双方k1*k2*H(id)的值来确定交集
            '''
            # 将Company和Partner的哈希值和索引构建DataFrame
            company_hash = pd.DataFrame([(ec0, i) for i, ec0 in enumerate(E_c_0)],columns=['hash','i'])
            partner_hash = pd.DataFrame([(ep0, j) for j, ep0 in enumerate(E_p_0)],columns=['hash','j'])
            # 找到交集的哈希值
            intersection = pd.merge(company_hash,partner_hash,how='inner',on='hash')
            return intersection

        # 在PYU设备上计算交集索引
        intersection = self.device(intersection_indices)(E_c_0, E_p_0)
        # 揭示交集结果（仅索引，不涉及原始ID）
        intersection = sf.reveal(intersection)
        # 生成随机数用于掩码Partner的特征。理论上随机数的范围应是Paillier的明文空间，但实际上小一些的值也不影响结果的正确性
        r_p = self.device(np.random.randint)(-out_dom // 2, out_dom // 2, size=(len(intersection), partner_data_shape[1]))
        # 将随机数加密
        r_p_enc = r_p.to(self.partner_heu).encrypt()

        print("Computing masked partner cipher")
        # 构建元组L：包含Company侧的交集索引和掩码后的Partner特征密文
        L = (
            intersection['i'].tolist(), 
            # 同态减法：从加密的Partner特征中减去加密的随机数掩码
            E_p_1[intersection['j'].tolist()] - r_p_enc,
        )
        # 筛选Partner公开特征的交集部分
        E_p_2 = sf.reveal(E_p_2)
        E_p_2 = E_p_2[intersection['j'].tolist()] if E_p_2 is not None else None
        print("Computing company shares")
        # 构建Company的秘密共享分片：将加密的Company特征解密后与随机数拼接
        R_cI = self.device(np.hstack)((
            E_c_1[intersection['i'].tolist()].to(self.device),
            r_p
        ))
        # 筛选Company公开特征的交集部分
        E_c_2 = sf.reveal(E_c_2)
        E_c_2 = E_c_2[intersection['i'].tolist()] if E_c_2 is not None else None

        # 如果双方都有公开特征，则水平拼接作为分桶标签
        bucket_labels = np.hstack((E_c_2,E_p_2)) if E_p_2 is not None and E_c_2 is not None else None
        return (L, R_cI, bucket_labels)


# PSIPartner类：Partner方的PSI实现
class PSIPartner(PSIWorker):
    """
    PSIPartner继承自PSIWorker，代表Partner方的PS3I实现。
    Partner方接收Company的哈希值，进行二次乘方后返回，同时处理自己的数据。
    """
    def exchange(self,U_c : tuple[PYUObject, HEUObject, PYUObject], company_data_shape : tuple[int, int]):  
        """
        Partner方执行数据交换，接收Company的数据并进行处理。
        对Company的哈希值进行二次乘方，同时生成随机掩码保护Company的特征。
        ## Args:
         - U_c: Company发送的数据，包含乘方后的key哈希值、加密后的私有特征和公开特征
         - company_data_shape: Company私有特征的数据形状
        ## Returns:
         - E_c: 处理后的Company数据，包含二次乘方后的keys哈希值、掩码后的私有特征分片和公开特征
         - U_p: Partner的数据，包含乘方后的keys哈希值、加密后的私有特征和公开特征
         - data_shape: Partner私有特征的数据形状
        """
        # 调用父类方法获取Partner方处理后的数据
        U_p_0, U_p_1, U_p_2 = self.hash_raw()

        # 解包Company发送过来的数据
        U_c_0, U_c_1, U_c_2 = U_c

        # 生成随机数用于掩码Company的特征
        self.r_c = self.device(np.random.randint)(- out_dom // 2, out_dom // 2, size=company_data_shape)

        print("Computing masked company cipher")
        # 定义标量乘法函数，用于计算Company数据的二次乘方
        def mul_k(U_c_0, k):
            return [crypto_scalarmult_ristretto255(k,u_c_0_i) for u_c_0_i in U_c_0]
        # 对Company的哈希值乘以Partner的私钥，得到二次乘方后的哈希值
        E_c_0 = self.device(mul_k)(U_c_0, self.k)
        # 将随机数加密用于同态减法
        r_c_enc = self.r_c.to(self.company_heu).encrypt()
        # 同态减法：从Company加密特征中减去随机数掩码
        E_c_1 = U_c_1 - r_c_enc
        # 公开特征保持不变
        E_c_2 = U_c_2
      
        # 定义重排函数，打乱Company数据的顺序以保护隐私
        def repermute(E_c_0, E_c_2, r_c):
            # 生成随机排列索引
            pem = np.random.permutation(company_data_shape[0]).tolist()
            # 按随机索引重排哈希值
            E_c_0 = [E_c_0[i] for i in pem]
            # 如果存在公开特征，也按随机索引重排
            if E_c_2 is not None:
                E_c_2 = E_c_2[pem]
            # 按随机索引重排随机数
            r_c = r_c[pem]
            return E_c_0, E_c_2, r_c, pem
        # 在PYU设备上执行重排操作
        E_c_0, E_c_2, self.r_c, pem = self.device(repermute,num_returns=4)(E_c_0, E_c_2, self.r_c)

        # 按相同的随机索引重排加密特征
        E_c_1 = E_c_1[pem]
        # 返回处理后的数据：E_c发送给Company，U_p也发送给Company用于交集计算
        return (E_c_0.to(self.company), E_c_1, E_c_2.to(self.company)), (U_p_0.to(self.company), U_p_1.to(self.partner_heu).encrypt(), U_p_2.to(self.company)), self.data_shape

    def output_shares(self, L):
        """
        输出Partner的秘密共享分片。
        根据交集索引提取对应的随机数分片，并与解密后的特征分片拼接。
        ## Args:
            L: 包含交集索引和掩码后的Partner特征密文的元组
        ## Returns:
            R_pI: Partner的秘密共享分片，由随机数分片和解密后的特征分片水平拼接而成
        """

        print("Computing partner shares")
        # 根据交集索引提取对应的随机数（Company特征的分片）
        r_c = self.device(np.ndarray.__getitem__)(self.r_c,L[0])
        # 构建Partner的秘密共享分片：随机数分片 + 解密后的Partner特征分片
        R_pI = self.device(np.hstack)((
            r_c,
            L[1].to(self.device)
        ))
        return R_pI

# 主函数：执行隐私求交集操作
def private_set_intersection(company_data : PYUObject, partner_data : PYUObject, heu_devices : tuple[HEU, HEU]) -> tuple[PYUObject, PYUObject,np.ndarray | None]:
    """
    执行私有集合交集（PS3I）操作的主函数。
    该函数协调Company和Partner两方进行数据交换和交集计算，最终生成秘密共享分片。
    
    ## Args:
     - company_data: Company持有的数据，应为包含键值、私有特征和公开特征的三元组
     - partner_data: Partner持有的数据，应为包含键值、私有特征和公开特征的三元组
    键值应处理为`list[str]`类型，私有特征应为32位浮点数`np.ndarray`，公开特征应为`None`或整形`np.ndarray`。
    
    ## Returns:
     - R_cI: Company的秘密共享分片（包含Company和Partner的特征分片）
     - R_pI: Partner的秘密共享分片（包含Company和Partner的特征分片）
     - buckets_labels: 分桶标签（如有），即Company和Partner的公开特征
    
    注意：运行PSI之后Company特征在共享分片左侧，Partner特征在共享分片右侧
    """
    # 获取双方的PYU设备
    pyu_devices = (company_data.device, partner_data.device)
    # 创建Company和Partner的PSI工作器实例
    psi_company = PSICompany(company_data, pyu_devices, heu_devices)
    psi_partner = PSIPartner(partner_data, pyu_devices, heu_devices)
    # 步骤1：Company发起数据交换，将哈希后的keys和加密特征发送出去
    U_c, company_data_shape = psi_company.exchange()
    # 步骤2：Partner接收Company数据，进行二次乘方和掩码处理，同时发送自己的数据
    E_c, U_p, partner_data_shape = psi_partner.exchange(U_c, company_data_shape)
    # 步骤3：Company计算交集，生成Company的秘密共享分片和Partner的掩码数据
    L, R_cI, buckets_labels = psi_company.compute_intersection(E_c, U_p, partner_data_shape)
    # 步骤4：Partner根据交集索引生成自己的秘密共享分片
    R_pI = psi_partner.output_shares(L)
    # 返回双方的秘密共享分片和分桶标签
    return R_cI, R_pI, buckets_labels

# ==================== 测试代码 ====================
# 下面的代码是生成随机数据集测试PSI性能，直接运行本文件即可

def generate_random_data(num_records, num_features):
    """
    生成随机测试数据。
    ## Args:
     - num_records: 记录数量
     - num_features: 特征数量
    ## Returns:
     - data: 包含随机keys和特征的DataFrame
    """
    # 导入随机数和字符串模块
    import random,string

    # 生成20位随机字母数字组合作为keys
    keys = [''.join(random.choices(string.ascii_uppercase +
                             string.digits, k=20)) for _ in range(num_records)]
    # 将keys转换为DataFrame，生成0-100之间的随机特征值
    keys = pd.DataFrame(keys)
    data = pd.DataFrame(100*np.random.rand(num_records,num_features))
    # 将keys和特征拼接
    data = pd.concat([keys,data],axis=1)
    # 重命名列索引
    data.columns = range(data.shape[1])
    return data

def test_PSI(company_data : pd.DataFrame, partner_data : pd.DataFrame):
    """
    测试PSI函数的执行。
    ## Args:
     - company_data: Company方的测试数据
     - partner_data: Partner方的测试数据
    """
    # 记录开始时间
    t1 = time()
    # 导入MPC初始化器
    from common import MPCInitializer
    # 初始化MPC环境
    mpc_init = MPCInitializer()
    company, partner = mpc_init.company, mpc_init.partner
    heu_devices = (mpc_init.company_heu, mpc_init.partner_heu)
    # 将Company数据转换为PSI所需的格式：(keys列表, 特征数组, None)
    company_data = (company_data.iloc[:,0].to_list(),company_data.iloc[:,1:].to_numpy(dtype=np.float32), None)
    company_data = sf.to(company,company_data)
    # 将Partner数据转换为PSI所需的格式
    partner_data = (partner_data.iloc[:,0].to_list(),partner_data.iloc[:,1:].to_numpy(dtype=np.float32), None)
    partner_data = sf.to(partner, partner_data)
    # 执行PSI操作
    R_cI, R_pI, bucket_labels = private_set_intersection(company_data, partner_data, heu_devices)
    print("PSI time taken: ",time()-t1)
    # 揭示Company和Partner的秘密共享分片
    R_cI = sf.reveal(R_cI)
    R_pI = sf.reveal(R_pI)
    # 将双方分片相加得到最终结果
    R_I = R_cI + R_pI
    print(R_I)

def random_PSI_test():
    """
    随机PSI测试函数。
    如果测试数据不存在，则生成新的随机数据并保存；
    如果已存在，则直接读取已有数据进行测试。
    """
    # 导入操作系统模块
    import os
    # 检查测试数据目录是否存在
    if not os.path.exists("Datasets/PSI"):
        # 创建测试数据目录
        os.makedirs("Datasets/PSI")
        # 生成交集数据
        intersection = generate_random_data(5,5)
        # 将交集数据分割
        intersection_left = intersection.iloc[:,:3]
        intersection_right = intersection.iloc[:,3:]
        # Partner的交集数据需要包含key列
        intersection_right = pd.concat([intersection.iloc[:,0],intersection_right],axis=1)
        # 生成Company独有的数据
        company_data = generate_random_data(10,2)
        # 将Company数据与交集数据拼接并随机打乱
        company_data = pd.concat([company_data,intersection_left],axis=0).sample(frac=1)
        # 生成Partner独有的数据
        partner_data = generate_random_data(5,3)
        # 统一列名
        partner_data.columns = intersection_right.columns
        # 将Partner数据与交集数据拼接并随机打乱
        partner_data = pd.concat([partner_data,intersection_right],axis=0).sample(frac=1)
        # 计算预期的交集结果用于验证
        intersection = pd.merge(company_data,partner_data,how='inner',on=0)
        # 保存测试数据到CSV文件
        company_data.to_csv("Datasets/PSI/company_data.csv",index=False)
        partner_data.to_csv("Datasets/PSI/partner_data.csv",index=False)
        intersection.to_csv("Datasets/PSI/intersection.csv",index=False)
    else:
        # 从CSV文件读取已有的测试数据
        company_data = pd.read_csv("Datasets/PSI/company_data.csv")
        partner_data = pd.read_csv("Datasets/PSI/partner_data.csv")
        intersection = pd.read_csv("Datasets/PSI/intersection.csv")
    # 提取交集的特征部分（去除key列）
    intersection = intersection.iloc[:,1:].to_numpy()
    print(intersection)
    # 执行PSI测试
    test_PSI(company_data, partner_data)

if __name__ == "__main__":
    random_PSI_test()
