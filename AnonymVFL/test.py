import os
import pandas as pd
import numpy as np
import tenseal as ts
from common import *
from LR import LR_test
from SVM import SVM_test
from PSI import PSICompany, PSIPartner, random_PSI_test

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

from sklearn.model_selection import train_test_split
from random import choices
import string
def PSI_LR():
    # data preprocessing
    data = pd.read_csv(os.path.join("Datasets","pcs.csv"),delimiter=',').astype(np.uint64)
    train_data, test_data = train_test_split(data)
    train_data.index = range(train_data.shape[0])
    keys = [''.join(choices(string.ascii_uppercase +
                             string.digits, k=20)) for _ in range(train_data.shape[0])]
    keys = pd.DataFrame(keys,columns=[0])
    train_data = pd.concat([keys,train_data],axis=1)


    train_data.columns = range(train_data.shape[1])
    labels = train_data.columns.tolist()
    company_data = train_data.sample(frac=0.9)
    company_data = company_data[labels[:5]]
    partner_data = train_data.sample(frac=0.9)
    partner_data = partner_data[[labels[0]]+labels[5:]]

    company = PSICompany(company_data)
    partner = PSIPartner(partner_data)
    U_c = company.exchange()
    E_c, U_p = partner.exchange(U_c)
    L, R_cI = company.compute_intersection(E_c, U_p)
    R_pI = partner.output_shares(L)

    R_cI = np.array(R_cI,dtype=np.int64) - out_dom
    R_pI = np.array(R_pI,dtype=np.int64)

    train_X = SharedVariable(R_cI[:,:-1],R_pI[:,:-1])
    train_y = SharedVariable(R_cI[:,-1].reshape(-1,1),R_pI[:,-1].reshape(-1,1))
    test_X = test_data.to_numpy()[:,:-1]
    test_y = test_data.to_numpy()[:,-1].reshape(-1,1)
    from LR import train
    weight = train(train_X, train_y, test_X, test_y).reveal()



if __name__ == "__main__":
    # random_PSI_test()
    # LR_test()
    # random_mul_test()
    for _ in range(100):
        mul_test()
        LT_test()
    # LR_test()
    # SVM_test()
    # PSI_LR()