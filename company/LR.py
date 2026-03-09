from SharedVariable import SharedVariable
import numpy as np
from tqdm import trange, tqdm
from common import out_dom
import jax.numpy as jnp
import secretflow as sf
from secretflow.device import SPUObject, PYUObject
from secretflow import SPU, PYU
from secretflow.data import FedNdarray, PartitionWay
from secretflow.data.ndarray import load
from common import approx_sigmoid, sigmoid, softmax, load_dataset
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

    def _forward(self, X : SPUObject) -> SPUObject | PYUObject:
        """
        ## Args:
         - X: 输入秘密共享的特征矩阵
        """
        def matmul(X, w):
            return X @ w
        z = self.spu(matmul)(X, self.w)
        z = z.to(self.train_label_keeper) # 将z发送给标签y的持有
        if self.approx:
            activate_fn = approx_sigmoid
        else:
            if self.out_features == 1:
                activate_fn = sigmoid
            else:
                activate_fn = softmax

        return self.train_label_keeper(activate_fn)(z)

    def dispatch_weight(self):
        assert isinstance(self.w, SPUObject), "Weights must be on SPU"
        def get_item(arr : jnp.ndarray, keys):
            return arr[keys]
        w1, w2= self.spu(get_item, static_argnames=['keys'])(self.w, np.arange(self.split_col)), self.spu(get_item, static_argnames=['keys'])(self.w, np.arange(self.split_col, self.in_features))
        w1 = w1.to(self.company)
        w2 = w2.to(self.partner)
        w = load({self.company: w1, self.partner: w2}, partition_way=PartitionWay.HORIZONTAL)
        return w

    def predict(self, X : FedNdarray, device : PYU, threshold: float = 0.33)  -> PYUObject:
        """
        ## Args:
         - X: 输入纵向划分的特征矩阵
         - device: 预测结果存放的PYU设备
         - threshold: 分类阈值，默认0.5，对于不平衡数据可以调低
        """
        assert isinstance(X, FedNdarray), "X must be a FedNdarray"
        assert isinstance(device,PYU), 'predictions must be moved to a PYU device'

        if isinstance(self.w, SPUObject):
            w = self.dispatch_weight()
        elif isinstance(self.w, FedNdarray):
            w = self.w
        z1 = self.company(lambda X, w: X @ w)(X.partitions[self.company], w.partitions[self.company]).to(device)
        z2 = self.partner(lambda X, w: X @ w)(X.partitions[self.partner], w.partitions[self.partner]).to(device)
        
        if self.approx:
            activate_fn = approx_sigmoid
        else:
            if self.out_features == 1:
                activate_fn = sigmoid
            else:
                activate_fn = softmax
            
        y = device(lambda a, b: activate_fn(a + b))(z1, z2)

        def to_int_labels(logits : np.ndarray, threshold):
            #将logit转化为整数标签，使用自定义阈值
            if logits.shape[1] == 1:
                return (logits > threshold).astype(int)
            else:
                return np.argmax(logits, axis=1)
        y = device(to_int_labels, static_argnames=['threshold'])(y, threshold)

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

    def fit(self, X : SPUObject, y : SPUObject | PYUObject, X_test : FedNdarray | None = None, y_test : PYUObject | None = None, batch_size = 64, val_steps = 1, n_epochs = 10, lr = 0.1, split_col : int = None):
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
        - split_col: 划分company特征和partner特征的列。左侧是company的特征，右侧是partner的特征。如未提供验证集则必须提供此项。
        ## Returns:
        - accs: 如果提供了验证集，则返回每次在验证集上评估的准确率。

        注意：训练完成后，self.w以明文的形式存储，company和partner各自持有一部分。推理时双方分别将各自的特征与各自的w相乘，然后由标签持有者聚合结果。
        """
        assert isinstance(X, SPUObject) and X.device == self.spu, "X must be on SPU"
        assert (isinstance(y, SPUObject) and y.device == self.spu) or (isinstance(y, PYUObject)), "y must be on active_party PYU or SPU"
        self.train_label_keeper = y.device
        num_samples, self.in_features = sf.reveal(self.spu(np.shape)(X))
        _, self.out_features = sf.reveal(self.train_label_keeper(np.shape)(y))
        self.in_features = int(self.in_features)
        self.out_features = int(self.out_features)
        self.w = np.zeros((self.in_features, self.out_features),dtype=np.float32)

        assert X_test is not None or split_col is not None, "Either validate set or split col must be provided"
        self.split_col = split_col if split_col is not None else sf.reveal(X_test.partition_shape()[self.company])[1]

        Xs = []
        ys = []
        validate = X_test is not None and y_test is not None
        if validate:
            assert isinstance(X_test, FedNdarray), "X_test must be a FedNdarray"
        if not self.approx:
            assert y.device != self.spu, "When approx is False, y must not be on SPU"
        for j in trange(0,num_samples,batch_size):
            batch = min(batch_size,num_samples - j)
            keys = np.arange(j, j + batch)
            def get_item(arr : jnp.ndarray, keys):
                return arr[keys]
            X_batch = self.spu(get_item, static_argnames=['keys'])(X, keys)
            y_batch = self.train_label_keeper(get_item, static_argnames=['keys'])(y, keys)
            Xs.append(X_batch)
            ys.append(y_batch)
        steps = 0
        accs = []
        f1s = []
        fOrs = []
        finalacc = 0
        finalf1 = 0
        finalfOr = 0
    
        for t in range(1,n_epochs + 1):
            print(f"Epoch {t}")
            for X,y in tqdm((zip(Xs, ys))):
                y_pred = self._forward(X)
                # 学习率随着迭代次数递减
                self._backward(X, y, y_pred, lr / t)
                if validate and steps % val_steps == 0:
                    y_pred = self.predict(X_test, y_test.device)
                    def compute_accuracy(y_true : np.ndarray, y_pred : np.ndarray):
                        y_true = y_true.reshape(-1,1)
                        y_pred = y_pred.reshape(-1,1)
                        # 调试输出：查看y_true和y_pred的实际值
                        # print(f"DEBUG - y_true unique values: {np.unique(y_true)}")
                        # print(f"DEBUG - y_pred unique values: {np.unique(y_pred)}")
                        # print(f"DEBUG - y_true shape: {y_true.shape}, y_pred shape: {y_pred.shape}")
                        # print(f"DEBUG - y_true first 100 values: {y_true[:100].flatten()}")
                        # print(f"DEBUG - y_pred first 100 values: {y_pred[:100].flatten()}")
                        return np.mean(y_true == y_pred)
                    def compute_f1(y_true : np.ndarray, y_pred : np.ndarray):
                        y_true = y_true.reshape(-1,1)
                        y_pred = y_pred.reshape(-1,1)
                        # print(f"DEBUG F1 - y_true range: [{np.min(y_true)}, {np.max(y_true)}]")
                        # print(f"DEBUG F1 - y_pred range: [{np.min(y_pred)}, {np.max(y_pred)}]")
                        # 使用更安全的比较方式，处理浮点数
                        tp = np.sum((np.abs(y_true - 0.0) < 1e-6) & (np.abs(y_pred - 0.0) < 1e-6))
                        fp = np.sum((np.abs(y_true - 1.0) < 1e-6) & (np.abs(y_pred - 0.0) < 1e-6))
                        fn = np.sum((np.abs(y_true - 0.0) < 1e-6) & (np.abs(y_pred - 1.0) < 1e-6))
                        # print(f"DEBUG F1 - TP: {tp}, FP: {fp}, FN: {fn}")
                        # F1分数
                        precision = tp / (tp + fp + 1e-8)  # 添加小数避免除零
                        recall = tp / (tp + fn + 1e-8)
                        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)  
                        return f1
                    
                    def compute_fOr(y_true : np.ndarray, y_pred : np.ndarray):
                        y_true = y_true.reshape(-1,1)
                        y_pred = y_pred.reshape(-1,1)
                        # print(f"DEBUG FOR - y_true range: [{np.min(y_true)}, {np.max(y_true)}]")
                        # print(f"DEBUG FOR - y_pred range: [{np.min(y_pred)}, {np.max(y_pred)}]")
                        tp = np.sum((y_true == 0) & (y_pred == 0))
                        fp = np.sum((y_true == 1) & (y_pred == 0))
                        fn = np.sum((y_true == 0) & (y_pred == 1))
                        tn = np.sum((y_true == 1) & (y_pred == 1))
                        # print(f"DEBUG FOR - TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
                        # 误漏率（False omission rate）指模型预测的全部阴性例数中实际患病者所占比例，反映了模型发现阴性者中患病的情况。
                        fOr = fn / (fn + tn + 1e-8)
                        return fOr
                    
                    acc = y_test.device(compute_accuracy)(y_test, y_pred)
                    f1 = y_test.device(compute_f1)(y_test, y_pred)
                    fOr = y_test.device(compute_fOr)(y_test, y_pred)
                    acc = sf.reveal(acc)
                    f1 = sf.reveal(f1)
                    fOr = sf.reveal(fOr)
                    accs.append(acc)
                    f1s.append(f1)
                    fOrs.append(fOr)
                    if t == n_epochs:
                        if acc > finalacc:
                            finalacc = acc
                        if f1 > finalf1:
                            finalf1 = f1
                        if fOr < finalfOr:
                            finalfOr = fOr
                    print(f"Step {steps}, Accuracy: {acc:.4f}, F1: {f1:.4f}, FOR: {fOr:.4f}")
                steps += 1

        self.w = self.dispatch_weight()
        print(f"\n📈 最终验证结果:")
        print(f"   • 最终准确率: {finalacc:.4f}")
        print(f"   • 最终F1分数: {finalf1:.4f}")
        print(f"   • 最终误漏率: {finalfOr:.4f}")
        return accs


    def save(self, paths : dict[str, str], ext = 'npy'):
        '''
        ## Args
        - paths: 保存模型的文件夹路径列表，包含company和partner的路径。例如：
        paths = {
            'company': 'path/to/company/model',
            'partner': 'path/to/partner/model'
        }
        '''
        assert isinstance(self.w, FedNdarray), "Weights must be a FedNdarray"
        w1, w2 = self.w.partitions[self.company], self.w.partitions[self.partner]
        info = {
            'shape' : (self.in_features, self.out_features),
            'lambda_': float(self.lambda_),
            'approx': bool(self.approx),
            'save_as' : ext
        }
        def save_model(w : np.ndarray, path : str):
            try:
                os.makedirs(path, exist_ok=True)
                print(f"Directory '{path}' created or already exists.")
            except OSError as e:
                print(f"Error creating directory '{path}': {e}")
            
            if ext == 'npy':
                np.save(os.path.join(path, 'weight.npy'), w)
            elif ext == 'csv':
                np.savetxt(os.path.join(path, 'weight.csv'), w, delimiter=',')
            json.dump(info, open(os.path.join(path, 'info.json'), 'w'))
        self.company(save_model)(w1, paths['company'])
        self.partner(save_model)(w2, paths['partner'])

    def load(self, paths):
        def load_model(path : str):
            info = json.load(open(os.path.join(path, 'info.json'), 'r'))
            ext = info['save_as']
            if ext == 'csv':
                w = np.loadtxt(os.path.join(path, 'weight.csv'), delimiter=',')
            else:
                w = np.load(os.path.join(path, 'weight.npy'))
            return w, info
        w1, info1 = self.company(load_model)(paths['company'])
        w2, info2 = self.partner(load_model)(paths['partner'])
        assert info1 == info2, "Model info mismatch between company and partner"
        self.w = load({self.company: w1, self.partner: w2}, partition_way=PartitionWay.HORIZONTAL)
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
    split_col = train_X.shape[1] // 2
    num_cat = train_y.shape[1] if len(train_y.shape) > 1 else 1
    test_X = load({company : sf.to(company, test_X[:, :split_col]), partner : sf.to(partner, test_X[:, split_col:])})
    test_y = sf.to(company, test_y)
    train_X = sf.to(company, train_X).to(spu)
    train_y = sf.to(company, train_y).to(spu)

    model = SSLR(devices, approx=True)
    accs = model.fit(train_X, train_y, X_test=test_X, y_test=test_y, n_epochs=10, batch_size=1024, val_steps=10, lr=0.1)
    model.save({
        'company': './company_model',
        'partner': './partner_model'
    },ext='csv')
    plt.plot(accs,label = "SSLR",color = "blue")

    test_X = np.hstack([sf.reveal(test_X.partitions[company]), sf.reveal(test_X.partitions[partner])])
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
    # SSLR_test("breast")
    SSLR_test("shop")