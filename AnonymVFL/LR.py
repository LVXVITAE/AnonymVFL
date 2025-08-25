from SharedVariable import SharedVariable
import numpy as np
from tqdm import trange, tqdm
from common import out_dom
import jax.numpy as jnp
import secretflow as sf
from secretflow.device import SPUObject, PYUObject
from secretflow import SPU, PYU
from secretflow.device.device.spu import SPUCompilerNumReturnsPolicy
from common import sigmoid, softmax, load_dataset, SS_share
import os, json

class SSLR:
    def __init__(self, devices: dict, lambda_ : float = 0, approx : bool = True):
        """
        ## Args: 
         - devices : 应包含四个字段，每个字段的值应为SPU或PYU。例如：

           devices = {
            'spu': spu,
            'company': company,
            'partner': partner,
            'coordinator': coordinator, # 如使用两方计算协议此字段可忽略
            'active_party': company
           }

         - lambda_: l2正则化参数，默认为0。
         - approx: 是否使用近似sigmoid函数，默认为True。
         这里提供了LR的两种实现。如果approx为True，则使用线性分段函数近似sigmoid函数。多分类场景下本模块会训练多个平行的2分类器（然而这样会导致每个分类器输出之和不为1，后续可考虑是否有更好的多分类算法）。
         如果approx为False，则不近似sigmoid函数。双方先用安全多方乘法计算z = X @ w的结果，再将z发送到y的持有者（y不作秘密共享），由y的持有者计算sigmoid和梯度。多分类场景下用本模块softmax函数代替sigmoid函数。
        """
        self.lambda_ = lambda_
        self.approx = approx
        assert 'spu' in devices and isinstance(devices['spu'], SPU), "devices must contain 'spu' of type SPU"
        self.spu = devices['spu']
        self.company = devices['company']
        self.partner = devices['partner']
        assert 'active_party' in devices and isinstance(devices['active_party'], PYU), "devices must contain 'active_party' of type PYU"
        self.active_party = devices['active_party']

    def _forward(self, X : SPUObject):
        """
        ## Args:
         - X: 输入秘密共享的特征矩阵
        """
        def matmul(X, w):
            return X @ w
        z = self.spu(matmul)(X, self.w)
        z = z.to(self.train_label_keeper) # 将z发送给标签y的持有
        if self.approx:
            def approx_sigmoid(z):
                return jnp.clip(z + 1/2, 0, 1)
            return self.train_label_keeper(approx_sigmoid)(z)
        else:
            if self.out_features == 1:
                return self.train_label_keeper(sigmoid)(z)
            else:
                return self.train_label_keeper(softmax)(z)

    def predict(self, X : SPUObject, device : PYU | None = None):
        """
        ## Args:
         - X: 输入秘密共享的特征矩阵
         - device: 预测结果存放的PYU设备
        """
        assert isinstance(X, SPUObject) and X.device == self.spu, "X must be on SPU"
        if device is None:
            device = self.active_party
        y = self._forward(X).to(device)
        assert isinstance(device,PYU), 'predictions must be moved to a PYU device'
        def to_int_labels(logits : jnp.ndarray):
            #将logit转化为整数标签
            if logits.shape[1] == 1:
                return jnp.round(logits)
            else:
                return jnp.argmax(logits, axis=1)
        y = device(to_int_labels)(y)

        return y

    def _backward(self, X : SPUObject, y : SPUObject | PYUObject, y_pred : SPUObject | PYUObject, lr : float = 0.1):
        """
        梯度下降步骤
        ## Args:
         - X: 输入秘密共享的特征矩阵
         - y: 标签，秘密共享或明文
         - y_pred: 模型预测结果，秘密共享或明文
         - lr: 梯度下降步长，默认为0.1
        """
        assert y.device == y_pred.device, "y and y_pred must be on the same device"
        def compute_gradient(y_pred, y):
            return y_pred - y
        grad = self.train_label_keeper(compute_gradient)(y_pred, y)
        grad = grad.to(self.spu)
        def grad_desc(lambda_, w : jnp.ndarray, X : jnp.ndarray, grad : jnp.ndarray):
            batch_size = X.shape[0]
            return (1 - lambda_) * w - (lr/batch_size) * (X.transpose() @ grad)
        self.w = self.spu(grad_desc)(self.lambda_, self.w, X, grad)

    def fit(self, X : SPUObject, y : SPUObject | PYUObject, X_test : SPUObject | None = None, y_test : PYUObject | None = None, batch_size = 64, val_steps = 1, n_epochs = 10, lr = 0.1):
        """
        训练指定轮数
        ## Args:
        - X: 训练集特征矩阵。
        - y: 训练集标签。
        - X_test: 验证集特征矩阵。如提供，将相隔若干step在验证集上评估准确率。
        - y_test: 验证集标签。
        - batch_size: 每个batch的样本数量，默认为64。
        - val_steps: 每隔多少个step在验证集上评估一次。每更新一次权重算一个step。默认为1。
        - n_epochs: 训练轮数，默认为10。
        - lr: 初始学习率，默认为0.1。学习率会随着迭代次数成反比。

        ## Returns:
        - accs: 如果提供了验证集，则返回每次在验证集上评估的准确率。
        """
        assert isinstance(X, SPUObject) and X.device == self.spu, "X must be on SPU"
        assert (isinstance(y, SPUObject) and y.device == self.spu) or (isinstance(y, PYUObject) and y.device == self.active_party), "y must be on active_party or SPU"
        self.train_label_keeper = y.device
        num_samples, self.in_features = sf.reveal(self.spu(jnp.shape)(X))
        _, self.out_features = sf.reveal(self.train_label_keeper(jnp.shape)(y))
        self.in_features = int(self.in_features)
        self.out_features = int(self.out_features)
        self.w = jnp.zeros((self.in_features, self.out_features),dtype=jnp.float32)

        Xs = []
        ys = []
        validate = X_test is not None and y_test is not None
        if validate:
            assert isinstance(X_test, SPUObject) and X_test.device == self.spu, "X and X_test must be on the same spu"
            assert isinstance(y_test, PYUObject) and y_test.device == self.active_party, "y_test must be on the active_party"
        if not self.approx:
            assert self.active_party != self.spu, "When approx is False, y must not be on SPU"
        for j in trange(0,num_samples,batch_size):
            batch = min(batch_size,num_samples - j)
            keys = jnp.arange(j, j + batch)
            def get_item(X : jnp.ndarray, keys):
                return X[keys]
            X_batch = self.spu(get_item, static_argnames=['keys'])(X, keys)
            y_batch = self.train_label_keeper(get_item, static_argnames=['keys'])(y, keys)
            Xs.append(X_batch)
            ys.append(y_batch)
        steps = 0
        accs = []
        for t in range(1,n_epochs + 1):
            print(f"Epoch {t}")
            for X,y in tqdm((zip(Xs, ys))):
                y_pred = self._forward(X)
                # 学习率随着迭代次数递减
                self._backward(X, y, y_pred, lr / t)
                if validate and steps % val_steps == 0:
                    y_pred = self.predict(X_test)
                    def compute_accuracy(y_true : jnp.ndarray, y_pred : jnp.ndarray):
                        y_true = y_true.reshape(-1,1)
                        y_pred = y_pred.reshape(-1,1)
                        return jnp.mean(y_true == y_pred)
                    acc = y_test.device(compute_accuracy)(y_test, y_pred)
                    acc = sf.reveal(acc)
                    accs.append(acc)
                    print(f"Iteration {steps}, Accuracy: {acc:.4f}")
                steps += 1

        return accs


    def save(self, paths : dict[str, str]):
        '''
        ## Args
        - paths: 保存模型的文件夹路径列表，包含company和partner的路径。例如：
        paths = {
            'company': 'path/to/company/model',
            'partner': 'path/to/partner/model'
        }
        '''
        w1, w2 = self.spu(SS_share,num_returns_policy=SPUCompilerNumReturnsPolicy.FROM_USER,user_specified_num_returns=2)(self.w)
        w1 = w1.to(self.company)
        w2 = w2.to(self.partner)
        info = {
            'shape' : (self.in_features, self.out_features),
            'lambda_': float(self.lambda_),
            'approx': bool(self.approx)
        }
        def save_model(w : jnp.ndarray, path : str):
            try:
                os.makedirs(path, exist_ok=True)
                print(f"Directory '{path}' created or already exists.")
            except OSError as e:
                print(f"Error creating directory '{path}': {e}")
            jnp.save(os.path.join(path, 'weight.npy'), w)
            json.dump(info, open(os.path.join(path, 'info.json'), 'w'))
        self.company(save_model)(w1, paths['company'])
        self.partner(save_model)(w2, paths['partner'])

    def load(self, paths):
        def load_model(path : str):
            w = jnp.load(os.path.join(path, 'weight.npy'))
            info = json.load(open(os.path.join(path, 'info.json'), 'r'))
            return w, info
        w1, info1 = self.company(load_model)(paths['company'])
        w2, info2 = self.partner(load_model)(paths['partner'])
        assert info1 == info2, "Model info mismatch between company and partner"
        w1 = w1.to(self.spu)
        w2 = w2.to(self.spu)
        self.w = self.spu(lambda a, b: a + b)(w1, w2)
        self.in_features, self.out_features = info1['shape']
        self.lambda_ = info1['lambda_']
        self.approx = info1['approx']

# 运行本文件直接执行这个函数
def SSLR_test(dataset):
    """
    （不执行PSI）测试SSLR性能，绘制损失曲线
    """
    from common import MPCInitializer
    mpc_init = MPCInitializer()
    spu = mpc_init.spu
    company = mpc_init.company
    partner = mpc_init.partner
    coordinator = mpc_init.coordinator
    devices = {
        'spu': spu,
        'company': company,
        'partner': partner,
        'coordinator': coordinator,
        'active_party': company
    }

    train_X, train_y, test_X, test_y = load_dataset(dataset)
    num_cat = train_y.shape[1] if len(train_y.shape) > 1 else 1
    test_X = sf.to(company, jnp.array(test_X)).to(spu)
    test_y = sf.to(company, jnp.array(test_y))
    train_X = sf.to(company, jnp.array(train_X)).to(spu)
    train_y = sf.to(company, jnp.array(train_y)).to(spu)

    model = SSLR(devices, approx=True)
    accs = model.fit(train_X, train_y, X_test=test_X, y_test=test_y, n_epochs=10, batch_size=1024, val_steps=10, lr=0.1)
    model.save({
        'company': './company_model',
        'partner': './partner_model'
    })
    plt.plot(accs,label = "SSLR",color = "blue")

    test_X = sf.reveal(test_X)
    test_y = sf.reveal(test_y)
    train_X = sf.reveal(train_X)
    train_y = sf.reveal(train_y)

    # 对比sklearn的LR实现
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(max_iter = 10,penalty=None)

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
    SSLR_test("breast")