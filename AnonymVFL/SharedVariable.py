import numpy as np
from common import VarCompany, VarPartner, share, out_dom

def is_constant(x):
    return isinstance(x, (int, float, np.ndarray))

def is_scalar(x):
    return isinstance(x, (int, float))

class PrivateData:
    def __init__(self, value : np.ndarray, owner):
        self.value = value
        self.owner = owner

    def shape(self):
        return self.value.shape  

    def __getitem__(self, key):
        return self.__class__(self.value[key], self.owner)  
    
    def __setitem__(self, key, val):
        assert isinstance(val, self.__class__), "Assigned value must be the same type"
        if self.owner == val.owner:
            self.value[key] = val.value
        else:
            raise ValueError("Owner mismatch")
    
    def send(self, receiver):
        return self.__class__(self.value, receiver)
    
    def __add__(self, other):
        assert isinstance(other, self.__class__), "Operands must be the same type"
        if self.owner == other.owner:
            return self.__class__(self.value + other.value, self.owner)
        else:
            raise ValueError("Owner mismatch")    
    
    def __sub__(self, other):
        assert isinstance(other, self.__class__), "Operands must be the same type"
        if self.owner == other.owner:
            return self.__class__(self.value - other.value, self.owner)
        else:
            raise ValueError("Owner mismatch")   
        
    def __mul__(self, other): 
        '''scalar or elementwise multiplication'''
        if isinstance(other, self.__class__):
            assert self.owner == other.owner, "Owner mismatch"
            return self.__class__(self.value * other.value, self.owner)
        elif is_constant(other):
            return self.__class__(self.value * other, self.owner)
        else:
            raise ValueError(f"Unsupported operand type(s) for *: '{self.__class__.__name__}' and '{type(other)}'")
        
    def __rmul__(self, other): # scalar or elementwise multiplication
        return self.__mul__(other)
    
    def __matmul__(self, other): # matrix multiplication
        if isinstance(other, self.__class__):
            assert self.owner == other.owner, "Owner mismatch"
            return self.__class__(self.value @ other.value, self.owner)
        elif is_constant(other):
            return self.__class__(self.value @ other, self.owner)
        else:
            raise ValueError(f"Unsupported operand type(s) for @: '{self.__class__.__name__}' and '{type(other)}'")
        
    def __rmatmul__(self, other): # matrix multiplication
        if isinstance(other, self.__class__):
            assert self.owner == other.owner, "Owner mismatch"
            return self.__class__(other.value @ self.value, self.owner)
        elif is_constant(other):
            return self.__class__(other @ self.value, self.owner)
        else:
            raise ValueError(f"Unsupported operand type(s) for @: '{type(other)}' and '{self.__class__.__name__}'")
        
    def __mod__(self, other : int):
        assert isinstance(other, int), "Modulo must be an integer"
        return self.__class__(np.fmod(self.value,other), self.owner)

class Share(PrivateData):
    pass

class SharedVariable:
    def __init__(self, share0 : Share, share1 : Share, company = VarCompany, partner = VarPartner):
        self.company = company
        self.partner = partner
        if not isinstance(share0, Share):
            share0 = Share(share0, company)
        if not isinstance(share1, Share):
            share1 = Share(share1, partner)

        assert share0.shape() == share1.shape(), "Shape mismatch"

        self.num_features = share0.shape()[1]
        # Ensure that share0 is from company and share1 is from partner
        if share0.owner == company and share1.owner == partner:
            self.share0 = share0
            self.share1 = share1
        elif share0.owner == partner and share1.owner == company:
            self.share0 = share1
            self.share1 = share0
        else:
            raise ValueError("Invalid owners")
    
    @classmethod
    def from_secret(cls, value, share_dom = out_dom, company = VarCompany, partner = VarPartner) -> 'SharedVariable':
        share0, share1 = share(value,share_dom)
        return cls(share0, share1, company, partner)
    
    @classmethod
    def zeroslike(cls, X, company = VarCompany, partner = VarPartner) -> 'SharedVariable':

        return cls.from_secret(np.zeros(X.shape(), dtype=np.int64), company = company, partner = partner)

    @classmethod
    def oneslike(cls, X, company = VarCompany, partner = VarPartner) -> 'SharedVariable':
        return cls.from_secret(np.ones(X.shape(), dtype=np.int64), company = company, partner = partner)
    
    @classmethod
    def random_like(cls, X, low, high, company = VarCompany, partner = VarPartner) -> 'SharedVariable':
        return cls.from_secret(np.random.randint(low, high, X.shape(), dtype=np.int64), company = company, partner = partner)

    def shape(self) -> tuple:
        return self.share0.shape()
    
    def reveal(self) -> np.ndarray:
        self.share1.send(self.company)
        return self.share0.value + self.share1.value
    
    def transpose(self) -> 'SharedVariable':
        return SharedVariable(self.share0.value.T, self.share1.value.T)
    

    def constant_conversion(self,x) -> 'SharedVariable': 
        '''
        Converts a constant to a SharedVariable.\n
        Scalars are expanded to the shape of the SharedVariable.
        '''
        if is_scalar(x):
            x = np.ones(self.shape()) * x
        if isinstance(x, np.ndarray):
            x = SharedVariable.from_secret(x, company = self.company, partner = self.partner)
        return x
    
    def __getitem__(self, key) -> 'SharedVariable':
        return SharedVariable(self.share0.__getitem__(key), self.share1.__getitem__(key))
    
    def __setitem__(self, key, val) -> None:
        if is_constant(val):
            val = self[key].constant_conversion(val)
        assert isinstance(val, SharedVariable), "Assigned value must be a SharedVariable"
        self.share0.__setitem__(key, val.share0)
        self.share1.__setitem__(key, val.share1)
        
    def __add__(self, other) -> 'SharedVariable':
        if is_constant(other):
            other = self.constant_conversion(other)
        assert isinstance(other, SharedVariable), "Unsupported operand type(s) for +: 'SharedVariable' and '{}'".format(type(other))
        return SharedVariable(self.share0 + other.share0, self.share1 + other.share1)
    
    def __radd__(self, other) -> 'SharedVariable':
        return self.__add__(other)
    
    def __sub__(self, other) -> 'SharedVariable':
        if is_constant(other):
            other = self.constant_conversion(other)
        assert isinstance(other, SharedVariable), "Unsupported operand type(s) for -: 'SharedVariable' and '{}'".format(type(other))
        return SharedVariable(self.share0 - other.share0, self.share1 - other.share1)
    
    def __rsub__(self, other) -> 'SharedVariable':
        if is_constant(other):
            other = self.constant_conversion(other)
        assert isinstance(other, SharedVariable), "Unsupported operand type(s) for -: '{}' and 'SharedVariable'".format(type(other))
        return SharedVariable(other.share0 - self.share0, other.share1 - self.share1)        
    
    def __mod__(self, other : int) -> 'SharedVariable':
        assert isinstance(other, int), "Modulo must be an integer"
        return SharedVariable(self.share0 % other, self.share1 % other)

    def mult_prepare(self, other):
        assert isinstance(other, SharedVariable), "Unsupported operand type(s) for * or @: 'SharedVariable' and '{}'".format(type(other))
        U = np.random.randint(0, out_dom, self.shape(),dtype=np.int64)
        V = np.random.randint(0, out_dom, other.shape(),dtype=np.int64)
        U0, U1 = share(U)
        U0 = Share(U0, self.company)
        U1 = Share(U1, self.partner)
        V0, V1 = share(V)
        V0 = Share(V0, self.company)
        V1 = Share(V1, self.partner)

        D0 = self.share0 - U0
        E0 = other.share0 - V0
        D0_ = D0.send(self.partner)
        E0_ = E0.send(self.partner)

        D1 = self.share1 - U1
        E1 = other.share1 - V1
        D1_ = D1.send(self.company)
        E1_ = E1.send(self.company)

        D = D0 + D1_
        E = E0 + E1_
        D_ = D0_ + D1
        E_ = E0_ + E1
        return U, U0, U1, V, V0, V1, D, D_, E, E_


    def __mul__(self, other) -> 'SharedVariable':
        if is_constant(other):
            return SharedVariable(self.share0 * other, self.share1 * other)
        
        U, U0, U1, V, V0, V1, D, D_, E, E_ = self.mult_prepare(other)

        Z = U * V

        Z0, Z1 = share(Z)
        Z0 = Share(Z0, self.company)
        Z1 = Share(Z1, self.partner)

        res0 = D * E + D * V0 + U0 * E + Z0
        res1 = D_ * V1 + U1 * E_ + Z1

        return SharedVariable(res0, res1)

    def __rmul__(self, other) -> 'SharedVariable':
        return self.__mul__(other)

    def __matmul__(self, other) -> 'SharedVariable':
        if is_constant(other):
            return SharedVariable(self.share0 @ other, self.share1 @ other)
        
        U, U0, U1, V, V0, V1, D, D_, E, E_ = self.mult_prepare(other)

        Z = U @ V

        Z0, Z1 = share(Z)
        Z0 = Share(Z0, self.company)
        Z1 = Share(Z1, self.partner)

        res0 = D @ E + D @ V0 + U0 @ E + Z0
        res1 = D_ @ V1 + U1 @ E_ + Z1

        return SharedVariable(res0, res1)
    
    def __rmatmul__(self, other) -> 'SharedVariable':
        if is_constant(other):
            return SharedVariable(other @ self.share0, other @ self.share1)
        elif isinstance(other, SharedVariable):
            return other @ self
        else:
            raise ValueError(f"Unsupported operand type(s) for @: '{type(other)}' and 'SharedVariable'")
    
    def cmp_prepare(self, other):

        if is_constant(other):
            other = self.constant_conversion(other)

        assert isinstance(other, SharedVariable), "Unsupported operand type(s) for comparison: 'SharedVariable' and '{}'".format(type(other))

        l = SharedVariable.oneslike(self) * 4
        z = self - other + l
        r0 = SharedVariable.random_like(self, low = 1, high = out_dom // 8)
        r1 = SharedVariable.random_like(self, low = 1, high = out_dom // 8)
        s = z * r0 + r1
        h = l * r0 + r1

        s0, s1 = s.share0, s.share1
        h0, h1 = h.share0, h.share1
        s1 = s1.send(self.company)
        h1 = h1.send(self.company)
        s = self.company.reconstruct(self.company,s0, s1)
        h = self.company.reconstruct(self.company,h0, h1)
        return s, h

    def __lt__(self,other) -> np.ndarray:
        s, h = self.cmp_prepare(other)
        return s < h

    def __gt__(self,other) -> np.ndarray:
        s, h = self.cmp_prepare(other)
        return s > h
    
    def __le__(self,other) -> np.ndarray:
        s, h = self.cmp_prepare(other)
        return s <= h
    
    def __ge__(self,other) -> np.ndarray:
        s, h = self.cmp_prepare(other)
        return s >= h    
