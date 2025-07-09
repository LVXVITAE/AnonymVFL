import numpy as np
import jax.numpy as jnp
import pandas as pd
import os
from sklearn.model_selection import train_test_split
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

def sigmoid(x : jnp.ndarray) -> jnp.ndarray:
    """
    Computes the sigmoid function.
    """
    return 1 / (1 + jnp.exp(-x))

def softmax(x : jnp.ndarray) -> jnp.ndarray:
    """
    Computes the softmax function.
    """
    e_x = jnp.exp(x - jnp.max(x, axis=-1, keepdims=True))
    return e_x / jnp.sum(e_x, axis=-1, keepdims=True)

def cross_entropy(y_true : jnp.ndarray, y_pred : jnp.ndarray) -> jnp.ndarray:
    """
    Computes the cross-entropy loss.
    """
    return -jnp.sum(y_true * jnp.log(y_pred + 1e-12))

class SigmoidCrossEntropy:
    @staticmethod
    def loss(y_true : jnp.ndarray, y_pred : jnp.ndarray) -> jnp.ndarray:
        """
        Computes the sigmoid cross-entropy loss.
        """
        y_pred = sigmoid(y_pred)
        return cross_entropy(y_true, y_pred)
    
    @staticmethod
    def grad(y_true : jnp.ndarray, y_pred : jnp.ndarray) -> jnp.ndarray:
        """
        Computes the gradient of the sigmoid cross-entropy loss.
        """
        y_pred = sigmoid(y_pred)
        return y_pred - y_true
    @staticmethod
    def hess(y_true : jnp.ndarray, y_pred : jnp.ndarray) -> jnp.ndarray:
        """
        Computes the hessian of the sigmoid cross-entropy loss.
        """
        y_pred = sigmoid(y_pred)
        return y_pred * (1 - y_pred)

class SoftmaxCrossEntropy:
    @staticmethod
    def loss(y_true : jnp.ndarray, y_pred : jnp.ndarray) -> jnp.ndarray:
        """
        Computes the softmax cross-entropy loss.
        """
        y_pred = softmax(y_pred)
        return cross_entropy(y_true, y_pred)
    
    @staticmethod
    def grad(y_true : jnp.ndarray, y_pred : jnp.ndarray) -> jnp.ndarray:
        """
        Computes the gradient of the softmax cross-entropy loss.
        """
        y_pred = softmax(y_pred)
        return y_pred - y_true
    
    @staticmethod
    def hess(y_true : jnp.ndarray, y_pred : jnp.ndarray) -> jnp.ndarray:
        """
        Computes the hessian of the softmax cross-entropy loss.
        """
        y_pred = softmax(y_pred)
        return y_pred * (1 - y_pred)
    
class MeanSquare:
    @staticmethod
    def loss(y_true : jnp.ndarray, y_pred : jnp.ndarray) -> jnp.ndarray:
        """
        Computes the mean square loss.
        """
        return jnp.mean((y_true - y_pred) ** 2)

    @staticmethod
    def grad(y_true : jnp.ndarray, y_pred : jnp.ndarray) -> jnp.ndarray:
        """
        Computes the gradient of the mean square loss.
        """
        return 2 * (y_pred - y_true) / y_true.shape[0]

    @staticmethod
    def hess(y_true : jnp.ndarray, y_pred : jnp.ndarray) -> jnp.ndarray:
        """
        Computes the hessian of the mean square loss.
        """
        return 2 / y_true.shape[0]
    
def load_dataset(dataset : str) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    if dataset == "pima" or dataset == "lbw" or dataset == "pcs" or dataset == "uis":
        data = pd.read_csv(os.path.join("Datasets",f"{dataset}.csv")).to_numpy()
        train_data, test_data = train_test_split(data,shuffle=False)
        train_X = train_data[:,:-1]
        train_y = train_data[:,-1].reshape(-1,1)
        test_X = test_data[:,:-1]
        test_y = test_data[:,-1].reshape(-1,1)

    elif dataset == "gisette" or dataset == "arcene":
        folder = os.path.join("Datasets",dataset)
        train_X = np.loadtxt(os.path.join(folder,f"{dataset}_train.data"))
        train_y = np.loadtxt(os.path.join(folder,f"{dataset}_train.labels"))
        train_y[train_y == -1] = 0
        train_y = train_y.reshape(-1,1)
        test_X = np.loadtxt(os.path.join(folder,f"{dataset}_valid.data"))
        test_y = np.loadtxt(os.path.join(folder,f"{dataset}_valid.labels"))
        test_y[test_y == -1] = 0
        test_y = test_y.reshape(-1,1)

    elif dataset == "mnist":
        from sklearn.datasets import fetch_openml
        mnist = fetch_openml('mnist_784', version=1,as_frame=False)
        X = mnist.data
        y = mnist.target
        train_X, test_X, train_y, test_y = train_test_split(X, y,shuffle=False)
        train_y = train_y.astype(int).reshape(-1,1)
        from sklearn.preprocessing import OneHotEncoder
        train_y = OneHotEncoder().fit_transform(train_y).toarray()
        test_y = test_y.astype(int).reshape(-1,1)
    
    elif dataset == "risk":
        dir_path = os.path.join("Datasets", "data", "data")
        train = pd.read_csv(os.path.join(dir_path, "risk_assessment_all.csv"))
        test = pd.read_csv(os.path.join(dir_path, "risk_assessment_all_test.csv"))
        train_X = train.drop(columns=["id","y"]).to_numpy()
        train_y = train["y"].to_numpy().reshape(-1,1)
        test_X = test.drop(columns=["id","y"]).to_numpy()
        test_y = test["y"].to_numpy().reshape(-1,1)

    elif dataset == "breast":
        dir_path = os.path.join("Datasets", "data", "data")
        guest = pd.read_csv(os.path.join(dir_path, "breast_hetero_guest.csv"))
        host = pd.read_csv(os.path.join(dir_path, "breast_hetero_host.csv"))
        all = pd.concat([host, guest], join = 'inner', axis = 1)
        X = all.drop(columns=["id", "y"]).to_numpy()
        y = all["y"].to_numpy().reshape(-1,1)
        train_X, test_X, train_y, test_y = train_test_split(X, y, shuffle=True)

    return train_X, train_y, test_X, test_y