from common import  out_dom
import numpy as np
from hashlib import sha512
from rbcl import crypto_core_ristretto255_from_hash, crypto_core_ristretto255_scalar_random, crypto_scalarmult_ristretto255
import pandas as pd
from time import time
import secretflow as sf
from secretflow.device import PYUObject, HEUObject
from secretflow import HEU, PYU

class PSIWorker:
    """
    PSIWorker是PSICompany和PSIPartner的基类，提供了数据集读取和对本方持有数据进行加密的功能。
    """
    def __init__(self, data : PYUObject, pyu_devices: tuple[PYU, PYU], heu_devices : tuple[HEU,HEU])-> None:
        """
        ## Args:
         - keys: 参与方的唯一标识符，要求是一个字符串列表。
         - private_features: 参与方持有的私有特征数据，即需要加密的特征。
         - public_features: 参与方持有的公开特征数据，默认为None。这个参数是预留给XGBoost分桶标签使用。
        """
        self.device = data.device
        def unpack(data : tuple[list[str], np.ndarray, np.ndarray | None]):
            keys, private_features, public_features = data
            return keys, private_features, public_features
        self.keys, self.private_features, self.public_features = self.device(unpack, num_returns=3)(data)
        self.k = self.device(crypto_core_ristretto255_scalar_random)()
        self.data_shape = sf.reveal(self.device(np.shape)(self.private_features))
        self.company_heu, self.partner_heu = heu_devices
        self.company, self.partner = pyu_devices
        
    def hash_raw(self) -> tuple[PYUObject, HEUObject,PYUObject]:
        """
        对keys进行哈希处理，并将私有特征进行加密，最后key、私有特征和公开特征进行随机排列。
        ## Returns:
         - U_0: 哈希后的keys列表
         - U_1: 加密后的私有特征
         - U_2: 公开特征（如果存在）
        """
        U_0, U_1, U_2 = self.keys, self.private_features, self.public_features
        def repermute(U_0, U_1, U_2):
            pem = np.random.permutation(len(U_0)).tolist()
            U_0 = [U_0[i] for i in pem]
            U_1 = U_1[pem]
            if U_2 is not None:
                U_2 = U_2[pem]
            return U_0, U_1, U_2
        U_0, U_1, U_2 = self.device(repermute,num_returns=3)(U_0, U_1, U_2)

        def hash_mul_keys(keys, k):
            keys = [crypto_core_ristretto255_from_hash(sha512(key.encode()).digest()) for key in keys]
            return [crypto_scalarmult_ristretto255(k, key) for key in keys]
        U_0 = self.device(hash_mul_keys)(U_0, self.k)
        # 对U_0, U_1, U_2进行随机排列
        return (U_0, U_1, U_2)

class PSICompany(PSIWorker):
    def exchange(self):
        """
        ## Returns:
         - U_c: 包含乘方后的key哈希值、加密后的私有特征
        """
        U_c_0, U_c_1, U_c_2 = self.hash_raw()
        U_c_0 = U_c_0.to(self.partner)
        U_c_1 = U_c_1.to(self.company_heu).encrypt()
        U_c_2 = U_c_2.to(self.partner)
        return (U_c_0, U_c_1, U_c_2), self.data_shape
    def compute_intersection(self,E_c ,U_p, partner_data_shape : tuple[int, int]):
        """
        计算交集
        ## Args:
         - E_c: Company加密后的数据，包含二次乘方后的key哈希值、加密后的私有特征和公开特征
         - U_p: Partner加密后的数据，包含乘方后的keys哈希值、加密后的私有特征和公开特征
        ## Returns:
         - L: 交集的索引和未解密的Partner私有特征分片
         - R_cI: Company私有特征分片和公开特征。其中R_cI[0]是私有特征分片，R_cI[1]是公开特征。
         - bucket_labels: 分桶标签（如有），即Company和Partner的公开特征。
        """
        U_p_0, U_p_1, U_p_2 = U_p

        # 计算Partner二次乘方后的哈希值
        def mul_k(U_p_0, k):
            return [crypto_scalarmult_ristretto255(k,u_p_0_i) for u_p_0_i in U_p_0]
        E_p_0 = self.device(mul_k)(U_p_0, self.k)
        E_p_1, E_p_2 = U_p_1, U_p_2

        E_c_0, E_c_1, E_c_2 = E_c

        def intersection_indices(E_c_0, E_p_0):
            '''比较二次乘方后的哈希值求交集'''
            # 此处可考虑针对非平衡数据集场景进行优化
            company_hash = pd.DataFrame([(ec0, i) for i, ec0 in enumerate(E_c_0)],columns=['hash','i'])
            partner_hash = pd.DataFrame([(ep0, j) for j, ep0 in enumerate(E_p_0)],columns=['hash','j'])
            intersection = pd.merge(company_hash,partner_hash,how='inner',on='hash')
            return intersection

        intersection = self.device(intersection_indices)(E_c_0, E_p_0)
        intersection = sf.reveal(intersection)
        # 生成随机数。理论上随机数的范围应是Paillier的明文空间，但实际上小一些的值也不影响结果的正确性
        # 后续可研究如何获取Paillier的明文空间的值
        r_p = self.device(np.random.randint)(-out_dom // 2, out_dom // 2, size=(len(intersection), partner_data_shape[1]))
        r_p_enc = r_p.to(self.partner_heu).encrypt()

        print("Computing masked partner cipher")
        L = (
            intersection['i'].tolist(), 
            # homo sub
            E_p_1[intersection['j'].tolist()] - r_p_enc,
        )
        E_p_2 = sf.reveal(E_p_2)
        E_p_2 = E_p_2[intersection['j'].tolist()] if E_p_2 is not None else None
        print("Computing company shares")
        R_cI = self.device(np.hstack)((
            E_c_1[intersection['i'].tolist()].to(self.device),
            r_p
        ))
        E_c_2 = sf.reveal(E_c_2)
        E_c_2 = E_c_2[intersection['i'].tolist()] if E_c_2 is not None else None

        bucket_labels = np.hstack((E_c_2,E_p_2)) if E_p_2 is not None and E_c_2 is not None else None
        return (L, R_cI, bucket_labels)


class PSIPartner(PSIWorker):
    def exchange(self,U_c : tuple[PYUObject, HEUObject, PYUObject], company_data_shape : tuple[int, int]):  
        """
        交换数据和公钥
        ## Args:
         - U_c: Company加密后的数据，包含乘方后的key哈希值、加密后的私有特征和公开特征
        ## Returns:
         - E_c: Company加密后的数据，包含二次乘方后的keys哈希值、未解密的私有特征分片和公开特征
         - U_p: Partner乘方keys哈希值、加密后的私有特征和公开特征
        """
        U_p_0, U_p_1, U_p_2 = self.hash_raw()

        U_c_0, U_c_1, U_c_2 = U_c

        # 生成随机数。理论上随机数的范围应是Paillier的明文空间，但实际上小一些的值也不影响结果的正确性
        # 后续可研究如何获取Paillier的明文空间的值
        self.r_c = self.device(np.random.randint)(- out_dom // 2, out_dom // 2, size=company_data_shape)

        print("Computing masked company cipher")
        # 计算Company二次乘方后的哈希值
        def mul_k(U_c_0, k):
            return [crypto_scalarmult_ristretto255(k,u_c_0_i) for u_c_0_i in U_c_0]
        E_c_0 = self.device(mul_k)(U_c_0, self.k)
        r_c_enc = self.r_c.to(self.company_heu).encrypt()
        # homo sub
        E_c_1 = U_c_1 - r_c_enc
        E_c_2 = U_c_2
      
        def repermute(E_c_0, E_c_2, r_c):
            pem = np.random.permutation(company_data_shape[0]).tolist()
            E_c_0 = [E_c_0[i] for i in pem]
            if E_c_2 is not None:
                E_c_2 = E_c_2[pem]
            r_c = r_c[pem]
            return E_c_0, E_c_2, r_c, pem
        E_c_0, E_c_2, self.r_c, pem = self.device(repermute,num_returns=4)(E_c_0, E_c_2, self.r_c)

        E_c_1 = E_c_1[pem]
        return (E_c_0.to(self.company), E_c_1, E_c_2.to(self.company)), (U_p_0.to(self.company), U_p_1.to(self.partner_heu).encrypt(), U_p_2.to(self.company)), self.data_shape

    def output_shares(self, L):
        """
        输出Partner的私有特征分片和公开特征
        ## Args:
            L: 包含交集索引、未解密的Partner私有特征分片和公开特征
        ## Returns:
            R_pI: Partner的私有特征分片和公开特征。其中R_pI[0]是私有特征分片，R_pI[1]是公开特征。
        """

        print("Computing partner shares")
        r_c = self.device(np.ndarray.__getitem__)(self.r_c,L[0])
        R_pI = self.device(np.hstack)((
            r_c,
            L[1].to(self.device)
        ))
        return R_pI

def private_set_intersection(company_data : PYUObject, partner_data : PYUObject, heu_devices : tuple[HEU, HEU]) -> tuple[PYUObject, PYUObject,np.ndarray | None]:
    """
    执行私有集合交集操作
    ## Args:
     - company_data: Company持有的数据，应为包含键值、私有特征和公开特征的三元组
     - partner_data: Partner持有的数据，应为包含键值、私有特征和公开特征的三元组
    键值应处理为`list[str]`类型，私有特征应为32位浮点数`np.ndarray`，公开特征应为`None`或整形`np.ndarray`。
    ## Returns:
     - R_I: 交集结果，包含Company和Partner的私有特征分片和公开特征
    """
    pyu_devices = (company_data.device, partner_data.device)
    psi_company = PSICompany(company_data, pyu_devices, heu_devices)
    psi_partner = PSIPartner(partner_data, pyu_devices, heu_devices)
    U_c, company_data_shape = psi_company.exchange()
    E_c, U_p, partner_data_shape = psi_partner.exchange(U_c, company_data_shape)
    L, R_cI, buckets_labels = psi_company.compute_intersection(E_c, U_p, partner_data_shape)
    R_pI = psi_partner.output_shares(L)
    return R_cI, R_pI, buckets_labels

# 下面的代码是生成随机数据集测试PSI性能，直接运行本文件即可
def generate_random_data(num_records, num_features):
    import random,string

    keys = [''.join(random.choices(string.ascii_uppercase +
                             string.digits, k=20)) for _ in range(num_records)]
    keys = pd.DataFrame(keys)
    data = pd.DataFrame(100*np.random.rand(num_records,num_features))
    data = pd.concat([keys,data],axis=1)
    data.columns = range(data.shape[1])
    return data

def test_PSI(company_data : pd.DataFrame, partner_data : pd.DataFrame):
    t1 = time()
    from common import MPCInitializer
    mpc_init = MPCInitializer()
    company, partner = mpc_init.company, mpc_init.partner
    heu_devices = (mpc_init.company_heu, mpc_init.partner_heu)
    company_data = (company_data.iloc[:,0].to_list(),company_data.iloc[:,1:].to_numpy(dtype=np.float32), None)
    company_data = sf.to(company,company_data)
    partner_data = (partner_data.iloc[:,0].to_list(),partner_data.iloc[:,1:].to_numpy(dtype=np.float32), None)
    partner_data = sf.to(partner, partner_data)
    R_cI, R_pI, bucket_labels = private_set_intersection(company_data, partner_data, heu_devices)
    print("PSI time taken: ",time()-t1)
    R_cI = sf.reveal(R_cI)
    R_pI = sf.reveal(R_pI)
    R_I = R_cI + R_pI
    print(R_I)

def random_PSI_test():
    import os
    if not os.path.exists("Datasets/PSI"):
        os.makedirs("Datasets/PSI")
        intersection = generate_random_data(5,5)
        intersection_left = intersection.iloc[:,:3]
        intersection_right = intersection.iloc[:,3:]
        intersection_right = pd.concat([intersection.iloc[:,0],intersection_right],axis=1)
        company_data = generate_random_data(10,2)
        company_data = pd.concat([company_data,intersection_left],axis=0).sample(frac=1)
        partner_data = generate_random_data(5,3)
        partner_data.columns = intersection_right.columns
        partner_data = pd.concat([partner_data,intersection_right],axis=0).sample(frac=1)
        intersection = pd.merge(company_data,partner_data,how='inner',on=0)
        company_data.to_csv("Datasets/PSI/company_data.csv",index=False)
        partner_data.to_csv("Datasets/PSI/partner_data.csv",index=False)
        intersection.to_csv("Datasets/PSI/intersection.csv",index=False)
    else:
        company_data = pd.read_csv("Datasets/PSI/company_data.csv")
        partner_data = pd.read_csv("Datasets/PSI/partner_data.csv")
        intersection = pd.read_csv("Datasets/PSI/intersection.csv")
    intersection = intersection.iloc[:,1:].to_numpy()
    print(intersection)
    test_PSI(company_data, partner_data)

if __name__ == "__main__":
    random_PSI_test()