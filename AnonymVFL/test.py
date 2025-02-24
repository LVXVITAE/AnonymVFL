import os
import pandas as pd
import numpy as np
import tenseal as ts
from PSI import *
from hashlib import sha512
from time import time
from common import *
from LR import LR_test
from SVM import SVM_test
def generate_random_data(num_records, num_features):
    import random,string

    keys = [''.join(random.choices(string.ascii_uppercase +
                             string.digits, k=20)) for _ in range(num_records)]
    keys = pd.DataFrame(keys,columns=['key'])
    data = pd.DataFrame(np.random.randint(0,out_dom,size=(num_records,num_features),dtype=np.uint64))
    data = pd.concat([keys,data],axis=1)
    return data

def test_PSI(company_data_path, partner_data_path, INTER_ori):
    t1 = time()
    company = PSICompany(company_data_path)
    partner = PSIPartner(partner_data_path)
    U_c = company.exchange()
    E_c, U_p = partner.exchange(U_c)
    L, R_cI = company.compute_intersection(E_c, U_p)
    R_pI = partner.output_shares(L)
    print("PSI time taken: ",time()-t1)
    R_I = R_cI.copy()
    INTER_res = set()
    for i in range(len(R_cI)):
        H_I = sha512()
        for j in range(len(R_cI[i])):
            R_I[i][j] = (R_cI[i][j] + R_pI[i][j]) % out_dom
            H_I.update(str(R_I[i][j]).encode())
        INTER_res.add(H_I.hexdigest())
    print(np.array(R_I))
    assert INTER_ori == INTER_res, "Intersection is not correct"

def random_PSI_test(scale = 10):
    intersection = generate_random_data(scale,10)
    company_data = pd.concat([generate_random_data(scale//2,10),intersection],axis=0).sample(frac=1)
    partner_data = pd.concat([generate_random_data(scale//2,10),intersection],axis=0).sample(frac=1)
    company_data_path = os.path.join("example","random_company.csv")
    partner_data_path = os.path.join("example","random_partner.csv")
    company_data.to_csv(company_data_path,header=False,index=False)
    partner_data.to_csv(partner_data_path,header=False,index=False)
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
    test_PSI(company_data_path, partner_data_path, INTER_ori)

def secureMM(he : ts.Context, X : np.ndarray, Y : np.ndarray):
    X_enc = ts.ckks_tensor(he, X)
    Z1 = np.random.randint(0, out_dom, (X.shape[0],Y.shape[1]),dtype=np.int64)
    Z0 = X_enc @ Y - Z1
    Z0 = np.array(Z0.decrypt().tolist())
    assert np.allclose(X @ Y, Z0 + Z1), "Secure matrix multiplication failed!"
    return Z0, Z1

from SharedVariable import SharedVariable

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

if __name__ == "__main__":
    # random_PSI_test()
    # LR_test()
    # random_mul_test()
    for _ in range(100):
        mul_test()
        LT_test()
    LR_test()
    SVM_test()