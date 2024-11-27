import os
import pandas as pd
import numpy as np
from PSI import PSICompany, PSIPartner
from hashlib import sha512
out_dom = int(2**64)
def generate_random_data(num_records, num_features):
    import random,string

    keys = [''.join(random.choices(string.ascii_uppercase +
                             string.digits, k=20)) for _ in range(num_records)]
    keys = pd.DataFrame(keys,columns=['key'])
    data = pd.DataFrame(np.random.randint(0,100,size=(num_records,num_features)))
    data = pd.concat([keys,data],axis=1)
    return data
def random_test():
    from time import time
    intersection = generate_random_data(10,10)
    company_data = pd.concat([generate_random_data(5,10),intersection],axis=0).sample(frac=1)
    partner_data = pd.concat([generate_random_data(5,10),intersection],axis=0).sample(frac=1)
    company_data_path = os.path.join(os.curdir,"example","test_company.csv")
    partner_data_path = os.path.join(os.curdir,"example","test_partner.csv")
    company_data.to_csv(company_data_path,header=False,index=False)
    partner_data.to_csv(partner_data_path,header=False,index=False)
    intersection = pd.merge(company_data,partner_data,how='inner')
    intersection = intersection.to_numpy()[:,1:]
    intersection = np.append(intersection,intersection,axis=1)
    INTER_ori = set()
    for I in intersection:
        H_I = sha512()
        for i in I:
            H_I.update(str(i).encode())
        INTER_ori.add(H_I.hexdigest())
        
    t1 = time()
    company = PSICompany(company_data_path)
    partner = PSIPartner(partner_data_path)
    print(intersection)
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
    
    assert INTER_ori == INTER_res, "Intersection is not correct"


class Test:
    def test_random(self):
        # for _ in range(5):
        random_test()