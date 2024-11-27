from common import Participant
from lightphe import LightPHE
import numpy as np
from hashlib import sha512
from random import shuffle,randint
from rbcl import *
from joblib import Parallel, delayed
import pandas as pd
from tqdm import tqdm

out_dom = int(2**64)
keysize = 2048
precision = 5

def encrypt(phe: LightPHE, plaintext: list[int]) -> list:
    cipher = list(map(phe.encrypt, plaintext))
    return cipher

def decrypt(phe: LightPHE, cipher: list) -> list:
    plaintext = list(map(phe.decrypt, cipher))
    return plaintext

def homo_sub(phe: LightPHE, E: list, r: list[int]) -> list:
    r_cipher = encrypt(phe,r)
    return list(map(lambda v1, v2: v1 + -1 * v2, E,r_cipher))

class PSIParticipant(Participant):
    def __init__(self,raw_data_path:str) -> None:
        super().__init__(raw_data_path)
        self.phe = LightPHE(algorithm_name='Paillier',key_size=keysize,precision=precision)
        self.k = crypto_core_ristretto255_scalar_random()
        self.num_features = self.raw_data.shape[1] - 1
        self.num_records = self.raw_data.shape[0]
    def hash_enc_raw(self):
        def hash_enc_each_row(rd):
            U_i0 = crypto_core_ristretto255_from_hash(sha512(rd[0].encode()).digest())

            U_i0 = crypto_scalarmult_ristretto255(self.k,U_i0)
            U_c_i1 = encrypt(self.phe,rd[1:])
            return (U_i0,U_c_i1)
        print("Hashing keys and encrypting raw data")
        U = Parallel(n_jobs = -1)(delayed(hash_enc_each_row)(rd) for _, rd in tqdm(self.raw_data.iterrows()))

        shuffle(U)
        return U        

class PSICompany(PSIParticipant):
    def exchange(self):
        self.phe.export_keys(target_file='company_pubkey.json',public=True)
        U_c = self.hash_enc_raw()
        return U_c
    def compute_intersection(self,E_c ,U_p):
        self.peer_num_features = len(U_p[0][1])
        self.peer_num_records = len(U_p)
        
        E_p = [(crypto_scalarmult_ristretto255(self.k,u_p_i[0]),u_p_i[1]) for u_p_i in U_p]
        
        self.phe_p = LightPHE(algorithm_name='Paillier',key_file='partner_pubkey.json',precision=precision)
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
            E_c_i_plaintext = decrypt(self.phe, E_c_i) # list(map(self.phe.decrypt, E_c_i))
            E_c_i_plaintext = [E_c_ik % out_dom  for E_c_ik in E_c_i_plaintext]
            r_p_j = [r_p_jk % out_dom for r_p_jk in r_p_j]
            
            R_cI_i = E_c_i_plaintext
            R_cI_i.extend(r_p_j)
            return R_cI_i     
        print("Computing company shares")  
        R_cI = Parallel(n_jobs=-1)(delayed(compute_R_cI)(i,j) for i,j in tqdm(id_intersection))
        return L, R_cI
            
            


                    



        

        

class PSIPartner(PSIParticipant):
    def exchange(self,U_c):  
        self.peer_num_features = len(U_c[0][1]) 
        self.peer_num_records = len(U_c)
        self.phe.export_keys(target_file='partner_pubkey.json',public=True)
        U_p = self.hash_enc_raw()

        self.phe_c = LightPHE(algorithm_name='Paillier',key_file='company_pubkey.json',precision=precision)

        
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
            E_p_j_plaintext = decrypt(self.phe, E_p_j) # list(map(self.phe.decrypt, E_p_j))
            R_pI_i1 = [(E_p_jk - N_p) % out_dom for E_p_jk in E_p_j_plaintext]
            R_pI_i = R_pI_i0
            R_pI_i.extend(R_pI_i1)
            return R_pI_i
        print("Computing partner shares")
        R_pI = Parallel(n_jobs=-1)(delayed(compute_R_pI)(i, E_p_j)for i, E_p_j in tqdm(L))
        return R_pI
