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
            keys: 参与方的唯一标识符，要求是一个字符串列表。
            private_features: 参与方持有的私有特征数据，即需要加密的特征。
            public_features: 参与方持有的公开特征数据，默认为None。这个参数是预留给XGBoost分桶标签使用。
        """
        if isinstance(keys, pd.DataFrame):
            keys = keys.values.tolist()
        self.keys = keys
        if isinstance(private_features, pd.DataFrame):
            private_features = private_features.to_numpy()
        self.private_features = private_features.astype(np.float32)
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
            U_0: 哈希后的keys列表
            U_1: 加密后的私有特征
            U_2: 公开特征（如果存在）
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
            U_c: 包含乘方后的key哈希值、加密后的私有特征
            pk_buffer: Company公钥
        """
        pk_buffer = pickle.dumps(self.kit.public_key())
        U_c = self.hash_enc_raw()
        return U_c, pk_buffer
    def compute_intersection(self,E_c ,U_p, partner_pk):
        """
        计算交集
        ## Args:
            E_c: Company加密后的数据，包含二次乘方后的key哈希值、加密后的私有特征和公开特征
            U_p: Partner加密后的数据，包含乘方后的keys哈希值、加密后的私有特征和公开特征
            partner_pk: Partner的公钥
        ## Returns:
            L: 交集的索引和未解密的Partner私有特征分片
            R_cI: Company私有特征分片和公开特征。其中R_cI[0]是私有特征分片，R_cI[1]是公开特征。
        """
        U_p_0, U_p_1, U_p_2 = U_p
        peer_num_records, peer_num_features = U_p_1.shape
        # 计算Partner二次乘方后的哈希值
        E_p_0 = [crypto_scalarmult_ristretto255(self.k,u_p_0_i) for u_p_0_i in U_p_0]
        E_p_1, E_p_2 = U_p_1, U_p_2

        E_c_0, E_c_1, E_c_2 = E_c

        self.pkit = hnp.setup(pickle.loads(partner_pk))
        # 比较二次乘方后的哈希值求交
        # 此处可考虑针对非平衡数据集场景进行优化
        company_hash = pd.DataFrame([(ec0, i) for i, ec0 in enumerate(E_c_0)],columns=['hash','i'])
        partner_hash = pd.DataFrame([(ep0, j) for j, ep0 in enumerate(E_p_0)],columns=['hash','j'])
        intersection = pd.merge(company_hash,partner_hash,how='inner',on='hash')

        r_p = np.random.randint(0, out_dom, size=(len(intersection), peer_num_features))
        r_p_enc = self.pkit.encryptor().encrypt(self.pkit.array(r_p, encoder=encoder))

        print("Computing masked partner cipher")
        L = (
            intersection['i'].values.tolist(), 
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
            U_c: Company加密后的数据，包含乘方后的key哈希值、加密后的私有特征和公开特征
            company_pk: Company的公钥
        ## Returns:
            E_c: Company加密后的数据，包含二次乘方后的keys哈希值、未解密的私有特征分片和公开特征
            U_p: Partner乘方keys哈希值、加密后的私有特征和公开特征
            pk_buffer: Partner公钥
        """
        pk_buffer = pickle.dumps(self.kit.public_key())
        U_p = self.hash_enc_raw()

        self.kit_c = hnp.setup(pickle.loads(company_pk))

        U_c_0, U_c_1, U_c_2 = U_c
        peer_num_records, peer_num_features = U_c_1.shape
        self.r_c = np.random.randint(0, out_dom, size=U_c_1.shape)

        print("Computing masked company cipher")
        E_c_0 = [crypto_scalarmult_ristretto255(self.k,u_c_0_i) for u_c_0_i in U_c_0] 
        r_c_enc = self.kit_c.encryptor().encrypt(self.kit_c.array(self.r_c, encoder=encoder))
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
    test_PSI(company_data, partner_data, INTER_ori)

if __name__ == "__main__":
    random_PSI_test()