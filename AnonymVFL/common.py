import numpy as np
out_dom = int(2**16)
class VarOwner:
    def __init__(self):
        pass
    def reconstruct(self, x0, x1):
        assert x0.owner == self and x1.owner == self
        return (x0 + x1).value

class VarCompany(VarOwner):
    pass

class VarPartner(VarOwner):
    pass

def share(x, share_dom = out_dom):
    '''split x into two additive shares'''
    if isinstance(x, (int,float)):
        x_1 = np.random.randint(0, share_dom)
    elif isinstance(x, np.ndarray):
        x_1 = np.random.randint(0, share_dom, x.shape,dtype=np.int64)
    x_2 = x - x_1
    return x_1, x_2
