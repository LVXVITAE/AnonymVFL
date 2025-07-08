import numpy as np
import jax.numpy as jnp
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