from common import  out_dom
from lightphe import LightPHE
import numpy as np
from hashlib import sha512
from random import shuffle,randint
from rbcl import *
from joblib import Parallel, delayed
import pandas as pd
from tqdm import tqdm

from time import time

scheme = 'Paillier'

def encrypt(phe: LightPHE, plaintext: list[int]) -> list:
    cipher = list(map(phe.encrypt, plaintext))
    return cipher

def decrypt(phe: LightPHE, cipher: list) -> list:
    plaintext = list(map(phe.decrypt, cipher))
    return plaintext

def homo_sub(phe: LightPHE, E: list, r: list[int]) -> list:
    r_cipher = encrypt(phe,r)
    return list(map(lambda v1, v2: v1 + -1 * v2, E,r_cipher))

class PSIWorker:
    def __init__(self,data : pd.DataFrame)-> None:
        self.data = data
        self.phe = LightPHE(algorithm_name=scheme)
        self.k = crypto_core_ristretto255_scalar_random()
        self.num_features = self.data.shape[1] - 1
        self.num_records = self.data.shape[0]
    def hash_enc_raw(self):
        def hash_enc_each_row(row):
            U_i0 = crypto_core_ristretto255_from_hash(sha512(row[0].encode()).digest())

            U_i0 = crypto_scalarmult_ristretto255(self.k,U_i0)
            U_c_i1 = encrypt(self.phe,row[1:])
            return (U_i0,U_c_i1)
        print("Hashing keys and encrypting raw data")
        U = Parallel(n_jobs = -1)(delayed(hash_enc_each_row)(row) for _, row in tqdm(self.data.iterrows()))

        shuffle(U)
        return U        

class PSICompany(PSIWorker):
    def exchange(self):
        self.phe.export_keys(target_file='company_pubkey.json',public=True)
        U_c = self.hash_enc_raw()
        return U_c
    def compute_intersection(self,E_c ,U_p):
        self.peer_num_features = len(U_p[0][1])
        self.peer_num_records = len(U_p)
        
        E_p = [(crypto_scalarmult_ristretto255(self.k,u_p_i[0]),u_p_i[1]) for u_p_i in U_p]
        
        self.phe_p = LightPHE(algorithm_name=scheme,key_file='partner_pubkey.json')
        company_hash = pd.DataFrame([(ec[0], i) for i, ec in enumerate(E_c)],columns=['hash','i'])
        partner_hash = pd.DataFrame([(ep[0], j) for j, ep in enumerate(E_p)],columns=['hash','j'])
        intersection = pd.merge(company_hash,partner_hash,how='inner',on='hash')
        id_intersection = intersection[['i','j']].to_numpy()
        N_p = self.phe_p.cs.plaintext_modulo
        r_p = [[randint(0,N_p) for _ in range(self.peer_num_features)] for _ in range(self.peer_num_records)]

        def compute_L(i,j):
            E_p_j = E_p[j][1]
            r_p_j = r_p[j]
            E_p_j = homo_sub(self.phe_p, E_p_j, r_p_j)
            return (i,E_p_j)
        
        print("Computing masked partner cipher")
        L = Parallel(n_jobs=-1)(delayed(compute_L)(i,j) for i, j in tqdm(id_intersection))

        def compute_R_cI(i,j):
            E_c_i = E_c[i][1]
            r_p_j =r_p[j]
            E_c_i_plaintext = decrypt(self.phe, E_c_i)
            E_c_i_plaintext = [E_c_ik % out_dom  for E_c_ik in E_c_i_plaintext]
            r_p_j = [r_p_jk % out_dom for r_p_jk in r_p_j]
            
            R_cI_i = E_c_i_plaintext
            R_cI_i.extend(r_p_j)
            return R_cI_i     
        print("Computing company shares")  
        R_cI = Parallel(n_jobs=-1)(delayed(compute_R_cI)(i,j) for i,j in tqdm(id_intersection))
        return L, np.array(R_cI)
            

class PSIPartner(PSIWorker):
    def exchange(self,U_c):  
        self.peer_num_features = len(U_c[0][1]) 
        self.peer_num_records = len(U_c)
        self.phe.export_keys(target_file='partner_pubkey.json',public=True)
        U_p = self.hash_enc_raw()

        self.phe_c = LightPHE(algorithm_name=scheme,key_file='company_pubkey.json')

        
        N_c = self.phe_c.cs.plaintext_modulo
        self.r_c = [[randint(0,N_c) for _ in range(self.peer_num_features)] for _ in range(self.peer_num_records)]

        def compute_E_c(i,u_c_i):
            E_c_i0 = crypto_scalarmult_ristretto255(self.k,u_c_i[0])
            E_c_i1 = homo_sub(self.phe_c, u_c_i[1], self.r_c[i])
            return (E_c_i0,E_c_i1)
        
        print("Computing masked company cipher")
        E_c = Parallel(n_jobs=-1)(delayed(compute_E_c)(i, u_c_i) for i, u_c_i in enumerate(tqdm(U_c)))

        pem = np.random.permutation(self.peer_num_records)
        E_c_pem = [E_c[i] for i in pem]
        self.r_c = [self.r_c[i] for i in pem]

        return E_c_pem, U_p
    
    def output_shares(self, L):
        N_c = self.phe_c.cs.plaintext_modulo
        N_p = self.phe.cs.plaintext_modulo
        
        def compute_R_pI(i, E_p_j):
            R_pI_i0 = [(r_c_ik - N_c) % out_dom for r_c_ik in self.r_c[i]]
            E_p_j_plaintext = decrypt(self.phe, E_p_j)
            R_pI_i1 = [(E_p_jk - N_p) % out_dom for E_p_jk in E_p_j_plaintext]
            R_pI_i = R_pI_i0
            R_pI_i.extend(R_pI_i1)
            return R_pI_i
        print("Computing partner shares")
        R_pI = Parallel(n_jobs=-1)(delayed(compute_R_pI)(i, E_p_j)for i, E_p_j in tqdm(L))
        return np.array(R_pI)


def generate_random_data(num_records, num_features):
    import random,string

    keys = [''.join(random.choices(string.ascii_uppercase +
                             string.digits, k=20)) for _ in range(num_records)]
    keys = pd.DataFrame(keys,columns=['0'])
    data = pd.DataFrame(np.random.randint(0,100,size=(num_records,num_features),dtype=np.uint64))
    data = pd.concat([keys,data],axis=1)
    data.columns = range(data.shape[1])
    return data

def test_PSI(company_data, partner_data, INTER_ori):
    t1 = time()
    company = PSICompany(company_data)
    partner = PSIPartner(partner_data)
    U_c = company.exchange()
    E_c, U_p = partner.exchange(U_c)
    L, R_cI = company.compute_intersection(E_c, U_p)
    R_pI = partner.output_shares(L)
    print("PSI time taken: ",time()-t1)
    R_I = (R_cI + R_pI) % out_dom
    INTER_res = set()
    for R_I_i in R_I:
        H_I = sha512()
        for R_I_ij in R_I_i:
            H_I.update(str(R_I_ij).encode())
        INTER_res.add(H_I.hexdigest())
    print(np.array(R_I))
    assert INTER_ori == INTER_res, "Intersection is not correct"

def random_PSI_test(scale = 100):
    intersection = generate_random_data(scale,100)
    company_data = pd.concat([generate_random_data(scale//2,100),intersection],axis=0).sample(frac=1)
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