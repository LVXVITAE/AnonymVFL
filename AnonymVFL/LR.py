from SharedVariable import SharedVariable
import numpy as np
from tqdm import trange, tqdm
from common import out_dom
import jax.numpy as jnp
import secretflow as sf
from secretflow.device.device import SPUObject, PYUObject

from common import MPCInitializer, sigmoid, softmax, load_dataset

mpc_init = MPCInitializer()
company, partner, coordinator, spu = mpc_init.company, mpc_init.partner, mpc_init.coordinator, mpc_init.spu

label_holder = company

class SSLR:
    def __init__(self, in_features : int, out_features : int = 1, lambda_ : float = 0, approx : bool = True):
        """
        ## Args: 
         - in_features: 输入特征的数量
         - out_features: 输出特征的数量，默认为1
         - lambda_: l2正则化参数，默认为0
         - approx: 是否使用近似sigmoid函数，默认为True。
         这里提供了LR的两种实现。如果approx为True，则使用线性分段函数近似sigmoid函数。多分类场景下本模块会训练多个平行的2分类器（然而这样会导致每个分类器输出之和不为1，后续可考虑是否有更好的多分类算法）。
         如果approx为False，则不近似sigmoid函数。双方先用安全多方乘法计算z = X @ w的结果，再将z发送到y的持有者（y不作秘密共享），由y的持有者计算sigmoid和梯度。多分类场景下用本模块softmax函数代替sigmoid函数。
        """
        self.out_features = out_features
        # 初始化权重为0
        self.w = label_holder(jnp.zeros)((in_features, out_features), dtype=jnp.float32).to(spu)
        self.lambda_ = lambda_
        self.approx = approx

    def forward(self, X : SPUObject):
        """
        ## Args:
         - X: 输入秘密共享的特征矩阵
        """
        if self.approx:
            def fw(X, w):
                # 使用线性分段函数近似sigmoid函数
                return jnp.clip(X @ w + 1/2, 0, 1)
            return spu(fw)(X, self.w)
        else:
            def fw(X, w):
                return X @ w
            z = spu(fw)(X, self.w)
            z = z.to(label_holder) # 将z发送给标签y的持有者
            if self.out_features == 1:
                return label_holder(sigmoid)(z)
            else:
                return label_holder(softmax)(z)

    def predict(self, X : SPUObject):
        """
        ## Args:
         - X: 输入秘密共享的特征矩阵
        """
        y = self.forward(X)
        y = y.to(label_holder)

        #将logit转化为整数标签
        if self.out_features == 1:
            y = label_holder(jnp.round)(y)
        else:
            y = label_holder(jnp.argmax)(y, axis=1)

        return sf.reveal(y).reshape(-1, 1)

    def backward(self, X : SPUObject, y : SPUObject | PYUObject, y_pred : SPUObject | PYUObject, lr : float = 0.1):
        """
        梯度下降步骤
        ## Args:
         - X: 输入秘密共享的特征矩阵
         - y: 标签，秘密共享或明文
         - y_pred: 模型预测结果，秘密共享或明文
         - lr: 梯度下降步长，默认为0.1
        """
        if not self.approx:
            y = y.to(spu)
            y_pred = y_pred.to(spu)
        def grad_desc(lambda_, w, X, y_pred, y):
            batch_size = X.shape[0]
            diff = y_pred - y
            return (1 - lambda_) * w - (lr/batch_size) * (X.transpose() @ diff)
        self.w = spu(grad_desc)(self.lambda_, self.w, X, y_pred, y)

    def fit(self, Xs, ys, n_iter = 100, lr = 0.1):
        """
        训练指定轮数
        ## Args:
        - Xs: 输入特征矩阵列表，每个元素都是X的一个batch，X秘密共享
        - ys: 标签列表，每个元素都是y的一个batch，y秘密共享或明文
        - n_iter: 训练轮数，默认为100
        - lr: 学习率，默认为0.1
        """
        for t in range(1,n_iter + 1):
            print(f"Epoch {t}")
            for X,y in tqdm(zip(Xs, ys)):
                y_pred = self.forward(X)
                # 学习率随着迭代次数递减
                self.backward(X, y, y_pred, lr / t)

    def save(self, path):
        # TODO
        raise NotImplementedError("Save method not implemented for SSLR")
    
    def load(self, path):
        # TODO
        raise NotImplementedError("Load method not implemented for SSLR")
    
# 运行本文件直接执行这个函数
def SSLR_test(dataset):
    """
    （不执行PSI）测试SSLR性能，绘制损失曲线
    """
    train_X, train_y, test_X, test_y = load_dataset(dataset)
    test_X = jnp.array(test_X)
    test_X = sf.to(company, test_X).to(spu)
    model = SSLR(train_X.shape[1], train_y.shape[1], approx=False)
    num_samples = train_X.shape[0]
    num_cat = train_y.shape[1]

    # 数据分批处理
    batch_size = 1024
    Xs = []
    ys = []
    for j in trange(0,num_samples,batch_size):
        batch = min(batch_size,num_samples - j)
        X_batch = jnp.array(train_X[j:j+batch])
        X_batch = sf.to(company, X_batch).to(spu)
        y_batch = jnp.array(train_y[j:j+batch])
        y_batch = sf.to(company, y_batch).to(spu)
        Xs.append(X_batch)
        ys.append(y_batch)
    
    n_iter = 20
    accs = []
    max_acc = 0
    for t in range(1,n_iter + 1):
        print(f"Epoch {t}")
        for X,y in tqdm(zip(Xs, ys)):
            y_pred = model.forward(X)
            model.backward(X, y, y_pred, 0.1 / t)

        y_pred = model.predict(test_X)
        Accracy = accuracy_score(test_y, y_pred)
        if Accracy > max_acc:
            max_acc = Accracy
            print(f"Iteration {t}, Accuracy: {Accracy:.4f}")
        accs.append(Accracy)

    plt.plot(accs,label = "SSLR",color = "blue")

    test_X = sf.reveal(test_X)

    # 对比sklearn的LR实现
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(max_iter = n_iter,penalty=None)

    if num_cat > 1:
        train_y = train_y.argmax(axis=1)

    model.fit(train_X,train_y.ravel())
    y_pred = model.predict(test_X)
    Accracy = accuracy_score(test_y, y_pred)
    plt.axhline(Accracy, 0, len(accs), label="LR sklearn", color = "red",linestyle = "--")

    plt.xlabel("nIter")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title(f"SSLR_{dataset}")
    plt.savefig(f"SSLR_{dataset}.png")
    plt.close()

# 早期测试用，可忽略
class LR:
    def __init__(self, in_features, out_features = 1, lambda_ = 0, appx_sigmoid = False):
        self.out_features = out_features
        self.w = np.zeros((in_features,out_features))
        self.appx_sigmoid = appx_sigmoid
        self.lambda_ = lambda_

    def activate_fn(self,X : np.ndarray):
        if self.appx_sigmoid:
            X += 1/2
            return np.clip(X,0,1)
        else:
            X = np.clip(X,-500,500)
            if self.out_features ==1:
                return 1/(1 + np.exp(-X))
            return np.exp(X) / np.sum(np.exp(X), axis=1, keepdims=True)

    def forward(self, X : np.ndarray):
        return self.activate_fn(X @ self.w)
    
    def predict(self, X):
        y = self.activate_fn(X @ self.w).round()
        if self.out_features == 1:
            return y
        else:
            return y.argmax(axis=1).reshape(-1,1)

    def backward(self, X : np.ndarray, y : np.ndarray, y_pred : np.ndarray, lr = 0.1):
        batch_size = X.shape[0]
        diff = y_pred - y
        self.w = (1 - self.lambda_) * self.w - (lr/batch_size) * (X.transpose() @ diff)

    def fit(self, X : np.ndarray, y : np.ndarray, n_iter = 100, batch_size = 64):
        num_samples, num_features = X.shape
        for t in range(1,n_iter + 1):
            for j in range(0,num_samples,batch_size):
                batch = min(batch_size,num_samples - j)
                X_batch = X[j:j+batch]
                y_batch = y[j:j+batch]

                y_pred = self.forward(X_batch)
                self.backward(X_batch, y_batch, y_pred, lr = 0.1 / t)

#早期测试使用，可暂时忽略
class LRSS:
    def __init__(self, in_features, out_features = 1, lambda_ = 0):
        self.out_features = out_features
        self.w = SharedVariable(np.zeros((in_features,out_features)),np.zeros((in_features, out_features)))
        self.lambda_ = lambda_

    @staticmethod
    def activate_fn(X : SharedVariable):
        GT_idx = np.argwhere(X > 1/2)
        LT_idx = np.argwhere(X < -1/2)
        X += 1/2
        for i, j in GT_idx:
            X[i, j] = 1
        for i, j in LT_idx:
            X[i, j] = 0
        return X

    def forward(self, X : SharedVariable):
        return self.activate_fn(X @ self.w)
    
    def predict(self, X : np.ndarray):
        y = np.clip((X @ self.w.reveal()) + 1/2, 0, 1).round()
        if self.out_features == 1:
            return y
        else:
            return y.argmax(axis=1).reshape(-1,1)
    
    def backward(self, X : SharedVariable, y : SharedVariable, y_pred : SharedVariable, lr = 0.1):
        batch_size = X.shape()[0]
        diff = y_pred - y
        self.w = (1 - self.lambda_) * self.w - (lr/batch_size) * (X.transpose() @ diff)

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
# 早期测试用，可忽略
def train(train_X : SharedVariable, train_y : SharedVariable, test_X, test_y, n_iter = 100, batch_size = 64) -> SharedVariable:
    num_samples, num_features = train_X.shape()
    _, num_cat = train_y.shape()
    
    model = LRSS(num_features, num_cat)
    accs = []
    max_acc = 0
    for t in range(1,n_iter + 1):
        print(f"Epoch {t}")
        for j in trange(0,num_samples,batch_size):
            batch = min(batch_size,num_samples - j)
            X = train_X[j:j+batch]
            y = train_y[j:j+batch]

            y_pred = model.forward(X)
            model.backward(X, y, y_pred, lr = 0.1 / t)

            y_pred = model.predict(test_X)
            Accracy = accuracy_score(test_y, y_pred)
            if Accracy > max_acc:
                max_acc = Accracy
                print(f"Iteration {t}, Batch {j//batch_size + 1}, Accuracy: {Accracy:.4f}")
        accs.append(Accracy)

    plt.plot(accs,label = "LR_SS",color = "blue")
    plt.axhline(max_acc, 0, len(accs), label="Max LR_SS", color = "blue",linestyle = ":")

    train_X = train_X.reveal()
    train_y = train_y.reveal()

    model = LR(num_features, num_cat,appx_sigmoid=True)
    accs = []
    max_acc = 0

    for t in range(1,n_iter + 1):
        print(f"Epoch {t}")
        for j in trange(0,num_samples,batch_size):
            batch = min(batch_size,num_samples - j)
            X = train_X[j:j+batch]
            y = train_y[j:j+batch]

            y_pred = model.forward(X)
            model.backward(X, y, y_pred, lr = 0.1 / t)

            y_pred = model.predict(test_X)
            Accracy = accuracy_score(test_y, y_pred)
            if Accracy > max_acc:
                max_acc = Accracy
        accs.append(Accracy)
    
    plt.plot(accs,label = "LR_without_SS", color = "red",linestyle = "--")
    plt.axhline(max_acc, 0, len(accs), label="Max LR_without_SS", color = "red",linestyle = ":")

    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(max_iter = n_iter,penalty=None)

    if num_cat > 1:
        train_y = train_y.argmax(axis=1)

    model.fit(train_X,train_y.ravel())
    y_pred = model.predict(test_X)
    Accracy = accuracy_score(test_y, y_pred)
    plt.axhline(Accracy, 0, len(accs), label="LR sklearn", color = "green",linestyle = ":")

    plt.xlabel("nIter")
    plt.ylabel("Accuracy")
    plt.legend()

# 早期测试用，可忽略
def LR_test(dataset):
    train_X, train_y, test_X, test_y = load_dataset(dataset)

    train_X = SharedVariable.from_secret(train_X, out_dom)
    train_y = SharedVariable.from_secret(train_y, out_dom)
    train(train_X, train_y, test_X,test_y)
    plt.title(f"LR_{dataset}")
    plt.savefig(f"LR_{dataset}.png")
    plt.close()

if __name__ == "__main__":
    # LR_test("mnist")
    # for dataset in ["pima","pcs","uis","gisette","arcene"]:
    #     LR_test(dataset)
    SSLR_test("risk")