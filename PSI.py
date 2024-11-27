from common import Participant
from lightphe import LightPHE
import numpy as np
from hashlib import sha512
from random import shuffle,randint
from rbcl import *
from multiprocessing import Pool

out_dom = int(2**64)
keysize = 2048
precision = 5

def encrypt(phe: LightPHE, plaintext: list[int]) -> list:
    cipher = list(map(phe.encrypt, plaintext))
    return cipher

def decrypt(phe: LightPHE, cipher: list) -> list:
    plaintext = list(map(phe.decrypt, cipher))
    return plaintext

class PSIParticipant(Participant):
    def __init__(self,raw_data_path:str) -> None:
        super().__init__(raw_data_path)
        self.phe = LightPHE(algorithm_name='Paillier',key_size=keysize,precision=precision)
        self.k = crypto_core_ristretto255_scalar_random()
        self.num_features = self.raw_data.shape[1] - 1
        self.num_records = self.raw_data.shape[0]

class PSICompany(PSIParticipant):
    def exchange(self):
        self.phe.export_keys(target_file='company_pubkey.json',public=True)
        U_c = []
        for i, rd in self.raw_data.iterrows():
            U_c_i0 = crypto_core_ristretto255_from_hash(sha512(rd[0].encode()).digest())
            assert crypto_core_ristretto255_is_valid_point(U_c_i0), "Invalid point"
            U_c_i0 = crypto_scalarmult_ristretto255(self.k,U_c_i0)
            with Pool() as p:
                U_c_i1 = encrypt(self.phe,rd.tolist()[1:]) # list(p.map(self.phe.encrypt,rd.tolist()[1:]))
            U_c.append((U_c_i0,U_c_i1))

        shuffle(U_c)
        return U_c
    def compute_intersection(self,E_c ,U_p):
        self.peer_num_features = len(U_p[0][1])
        self.peer_num_records = len(U_p)
        L = []
        R_cI = []
        E_p = []
        for i, u_p_i  in enumerate(U_p):
            E_p_i0 = crypto_scalarmult_ristretto255(self.k,u_p_i[0])
            E_p_i1 = u_p_i[1]
            E_p.append((E_p_i0,E_p_i1))
        
        self.phe_p = LightPHE(algorithm_name='Paillier',key_file='partner_pubkey.json',precision=precision)

        id_intersection = [(i,j) for i in range(len(E_c)) for j in range(len(E_p)) if E_c[i][0] == E_p[j][0]]
        N_p = self.phe_p.cs.plaintext_modulo
        for i, j in id_intersection:
            E_c_i = E_c[i][1]
            E_p_j = E_p[j][1]
            r_p_j = [randint(0,N_p) for _ in range(self.peer_num_features)]
 
            r_p_j_cipher = encrypt(self.phe_p,r_p_j) # list(map(self.phe_p.encrypt, r_p_j))
            E_p_j = list(map(lambda v1, v2: v1 + -1 * v2, E_p_j,r_p_j_cipher))

            L.append((i,E_p_j))

            E_c_i_plaintext = decrypt(self.phe, E_c_i) # list(map(self.phe.decrypt, E_c_i))
            E_c_i_plaintext = [E_c_ik % out_dom  for E_c_ik in E_c_i_plaintext]
            r_p_j = [r_p_jk % out_dom for r_p_jk in r_p_j]
            
            R_cI_i = E_c_i_plaintext
            R_cI_i.extend(r_p_j)
            R_cI.append(R_cI_i)
        return L, R_cI
            
            


                    



        

        

class PSIPartner(PSIParticipant):
    def exchange(self,U_c):  
        self.peer_num_features = len(U_c[0][1]) 
        self.peer_num_records = len(U_c)
        self.phe.export_keys(target_file='partner_pubkey.json',public=True)
        U_p = []
        for i, rd in self.raw_data.iterrows():
            U_p_i0 = crypto_core_ristretto255_from_hash(sha512(rd[0].encode()).digest())
            U_p_i0 = crypto_scalarmult_ristretto255(self.k,U_p_i0)
            U_p_i1 = encrypt(self.phe, rd.tolist()[1:]) # list(map(self.phe.encrypt, rd.tolist()[1:]))
            U_p.append((U_p_i0,U_p_i1))

        shuffle(U_p)

        self.phe_c = LightPHE(algorithm_name='Paillier',key_file='company_pubkey.json',precision=precision)

        E_c = []
        N_c = self.phe_c.cs.plaintext_modulo
        self.r_c = [[randint(0,N_c) for _ in range(self.peer_num_features)] for _ in range(self.peer_num_records)]

        for i, u_c_i in enumerate(U_c):
            E_c_i0 = crypto_scalarmult_ristretto255(self.k,u_c_i[0])

            r_c_i_cipher = encrypt(self.phe_c, self.r_c[i]) # list(map(self.phe_c.encrypt, self.r_c[i]))       
            E_c_i1 = list(map(lambda v1, v2: v1 + -1*v2, u_c_i[1], r_c_i_cipher))
            E_c.append((E_c_i0,E_c_i1))

        pem = np.random.permutation(self.peer_num_records)
        E_c_pem = [E_c[i] for i in pem]
        self.r_c = [self.r_c[i] for i in pem]

        return E_c_pem, U_p
    
    def output_shares(self, L):
        R_pI = []
        N_c = self.phe_c.cs.plaintext_modulo
        N_p = self.phe.cs.plaintext_modulo
        for i, E_p_j in L:
            R_pI_i0 = [(r_c_ik - N_c) % out_dom for r_c_ik in self.r_c[i]]
            E_p_j_plaintext = decrypt(self.phe, E_p_j) # list(map(self.phe.decrypt, E_p_j))
            R_pI_i1 = [(E_p_jk - N_p) % out_dom for E_p_jk in E_p_j_plaintext]
            R_pI_i = R_pI_i0
            R_pI_i.extend(R_pI_i1)
            R_pI.append(R_pI_i)
        return R_pI
