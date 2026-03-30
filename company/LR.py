import numpy as np
from tqdm import trange, tqdm
import jax.numpy as jnp
# 导入SecretFlow框架
import secretflow as sf
from secretflow.device import SPUObject, PYUObject
from secretflow import PYU
from secretflow.data import FedNdarray, PartitionWay
from secretflow.data.ndarray import load
# 导入公共模块中的函数和基类
from common import approx_sigmoid, sigmoid, softmax, compute_accuracy, compute_f1_metric, compute_for_metric
# 导入安全共享机器学习基类
from common import SSML
import os, json
import matplotlib.pyplot as plt


def spu_matmul(X, w):
    """矩阵乘法 X @ w"""
    return X @ w


def spu_get_item(arr: jnp.ndarray, keys):
    """数组索引取值 arr[keys]"""
    return arr[keys]


def compute_gradient(y_pred, y):
    """计算梯度：预测值减去真实标签"""
    return y_pred - y


def grad_desc(lambda_, w: jnp.ndarray, X: jnp.ndarray, grad: jnp.ndarray, lr: float):
    """梯度下降更新权重，包含L2正则化"""
    batch_size = X.shape[0]
    return (1 - lambda_) * w - (lr / batch_size) * (X.transpose() @ grad)


def to_int_labels_with_threshold(logits: np.ndarray, threshold):
    """将logit转化为整数标签，使用自定义阈值"""
    if logits.shape[1] == 1:
        return (logits > threshold).astype(int)
    else:
        return np.argmax(logits, axis=1)


def xw_product(X, w):
    """计算X @ w，用于predict中company/partner分别计算"""
    return X @ w


def xw_sum_activate(a, b, activate_fn):
    """聚合两方结果并应用激活函数"""
    return activate_fn(a + b)


def lr_save_model(w: np.ndarray, path: str, ext: str, info: dict):
    """保存LR模型权重和元信息到指定路径"""
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


def lr_load_model(path: str):
    """从指定路径加载LR模型权重和元信息"""
    info = json.load(open(os.path.join(path, 'info.json'), 'r'))
    ext = info['save_as']
    if ext == 'csv':
        w = np.loadtxt(os.path.join(path, 'weight.csv'), delimiter=',', ndmin=2)
    else:
        w = np.load(os.path.join(path, 'weight.npy'))
    return w, info

class SSLR(SSML):
    """秘密共享逻辑回归模型，支持二分类和多分类任务"""
    
    def __init__(self, devices: dict, lambda_ : float = 1e-5, approx : bool = True):
        """
        初始化SSLR模型
        ## Args: 
         - devices : 每个字段的值应为SPU或PYU。例如：

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
        # 分类阈值，用于将概率转换为类别标签
        self.pred_threshold = 0.5
        super().__init__(devices)

    def _forward(self, X : SPUObject) -> SPUObject | PYUObject:
        """
        前向传播函数，计算模型预测值
        ## Args:
         - X: 输入秘密共享的特征矩阵
        """
        # 在SPU上执行矩阵乘法，计算线性组合z = X @ w
        z = self.spu(spu_matmul)(X, self.w)
        # 将z发送给标签y的持有者
        z = z.to(self.train_label_keeper) # 将z发送给标签y的持有
        # 根据配置选择激活函数
        if self.approx:
            # 使用近似sigmoid函数（适用于秘密共享场景）
            activate_fn = approx_sigmoid
        else:
            # 根据输出特征数选择激活函数
            if self.out_features == 1:
                # 二分类使用sigmoid
                activate_fn = sigmoid
            else:
                # 多分类使用softmax
                activate_fn = softmax

        # 在标签持有者设备上执行激活函数
        return self.train_label_keeper(activate_fn)(z)

    def dispatch_weight(self):
        '''权重分发函数，将SPU上的权重分发到company和partner两方，存储为FedNdarray格式'''
        # 断言权重必须在SPU上
        assert isinstance(self.w, SPUObject), "Weights must be on SPU"
        # 分别获取company和partner对应的权重部分
        w1, w2= self.spu(spu_get_item, static_argnames=['keys'])(self.w, np.arange(self.split_col)), self.spu(spu_get_item, static_argnames=['keys'])(self.w, np.arange(self.split_col, self.in_features))
        # 将权重发送到各自的参与方，并封装为FedNdarray格式
        w1 = w1.to(self.company)
        w2 = w2.to(self.partner)
        w = load({self.company: w1, self.partner: w2}, partition_way=PartitionWay.HORIZONTAL)
        return w

    def predict(self, X : FedNdarray, device : PYU)  -> PYUObject:
        """
        模型预测函数，对输入数据进行分类预测
        ## Args:
         - X: 输入纵向划分的特征矩阵
         - device: 预测结果存放的PYU设备
         - threshold: 分类阈值，默认0.5，对于不平衡数据可以调低
        """
        # 类型检查：输入必须是联邦数组
        assert isinstance(X, FedNdarray), "X must be a FedNdarray"
        # 类型检查：预测结果必须存放到PYU设备
        assert isinstance(device,PYU), 'predictions must be moved to a PYU device'

        # 如果权重在SPU上，先分发到各方
        if isinstance(self.w, SPUObject):
            w = self.dispatch_weight()
        elif isinstance(self.w, FedNdarray):
            w = self.w
        # company和partner计算其特征与权重的乘积
        z1 = self.company(xw_product)(X.partitions[self.company], w.partitions[self.company]).to(device)
        z2 = self.partner(xw_product)(X.partitions[self.partner], w.partitions[self.partner]).to(device)
        
        # 根据配置选择激活函数
        if self.approx:
            activate_fn = approx_sigmoid
        else:
            if self.out_features == 1:
                activate_fn = sigmoid
            else:
                activate_fn = softmax
        # 在目标设备上聚合两方的结果并应用激活函数
        y = device(xw_sum_activate)(z1, z2, activate_fn)

        # 将概率输出转换为整数标签
        y = device(to_int_labels_with_threshold, static_argnames=['threshold'])(y, self.pred_threshold)

        return y

    def _backward(self, X : SPUObject, y : SPUObject | PYUObject, y_pred : SPUObject | PYUObject, lr : float = 0.1):
        """
        反向传播函数，执行梯度下降更新权重
        ## Args:
         - X: 输入秘密共享的特征矩阵
         - y: 标签，秘密共享或明文
         - y_pred: 模型预测结果，秘密共享或明文
         - lr: 梯度下降步长，默认为0.1
        """
        # 确保y和y_pred在同一设备上
        assert y.device == y_pred.device, "y and y_pred must be on the same device"
        # 在标签持有者设备上计算梯度
        grad = self.train_label_keeper(compute_gradient)(y_pred, y)
        # 将梯度发送到SPU
        grad = grad.to(self.spu)
        # 在SPU上执行梯度下降更新权重
        self.w = self.spu(grad_desc)(self.lambda_, self.w, X, grad, lr)

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
        # 类型检查：训练数据X必须在SPU上
        assert isinstance(X, SPUObject) and X.device == self.spu, "X must be on SPU"
        # 类型检查：标签y可以在SPU或PYU上
        assert (isinstance(y, SPUObject) and y.device == self.spu) or (isinstance(y, PYUObject)), "y must be on active_party PYU or SPU"
        # 记录标签持有者设备
        self.train_label_keeper = y.device
        # 获取训练数据的样本数和特征数、标签的输出维度
        num_samples, self.in_features = sf.reveal(self.spu(np.shape)(X))
        _, self.out_features = sf.reveal(self.train_label_keeper(np.shape)(y))
        self.in_features = int(self.in_features)
        self.out_features = int(self.out_features)
        # 初始化权重为零矩阵
        self.w = np.zeros((self.in_features, self.out_features),dtype=np.float32)

        # 确保提供了验证集或特征划分列
        assert X_test is not None or split_col is not None, "Either validate set or split col must be provided"
        # 获取特征划分列的位置
        self.split_col = split_col if split_col is not None else sf.reveal(X_test.partition_shape()[self.company])[1]

        # 用于存储分批数据的列表
        Xs = []
        ys = []
        # 判断是否进行验证
        validate = X_test is not None and y_test is not None
        if validate:
            assert isinstance(X_test, FedNdarray), "X_test must be a FedNdarray"
        # 非近似模式下，标签不能在SPU上
        if not self.approx:
            assert y.device != self.spu, "When approx is False, y must not be on SPU"
        # 将数据按batch分割
        for j in trange(0,num_samples,batch_size):
            batch = min(batch_size,num_samples - j)
            keys = np.arange(j, j + batch)
            # 获取当前batch的特征数据
            X_batch = self.spu(spu_get_item, static_argnames=['keys'])(X, keys)
            # 获取当前batch的标签数据
            y_batch = self.train_label_keeper(spu_get_item, static_argnames=['keys'])(y, keys)
            Xs.append(X_batch)
            ys.append(y_batch)
        # 初始化训练步数计数器
        steps = 1
        # 用于记录评估指标的列表
        accs = []
        f1s = []
        fOrs = []
        # 记录最终评估指标
        finalacc = 0
        finalf1 = 0
        finalfOr = 0
    
        # 开始训练循环
        for t in range(1,n_epochs + 1):
            print(f"Epoch {t}")
            for X,y in tqdm((zip(Xs, ys))):
                # 执行前向传播
                y_pred = self._forward(X)
                # 学习率随着迭代次数递减
                self._backward(X, y, y_pred, lr / t)
                # 如果需要验证且达到验证步数，则在验证集上评估模型性能
                if validate and steps % val_steps == 0:
                    y_pred = self.predict(X_test, y_test.device)
                    # def compute_accuracy(y_true : np.ndarray, y_pred : np.ndarray):
                    #     y_true = y_true.reshape(-1,1)
                    #     y_pred = y_pred.reshape(-1,1)
                    #     # 调试输出：查看y_true和y_pred的实际值
                    #     # print(f"DEBUG - y_true unique values: {np.unique(y_true)}")
                    #     # print(f"DEBUG - y_pred unique values: {np.unique(y_pred)}")
                    #     # print(f"DEBUG - y_true shape: {y_true.shape}, y_pred shape: {y_pred.shape}")
                    #     # print(f"DEBUG - y_true first 100 values: {y_true[:100].flatten()}")
                    #     # print(f"DEBUG - y_pred first 100 values: {y_pred[:100].flatten()}")
                    #     return np.mean(y_true == y_pred)
                    # 在验证设备上计算各项指标
                    acc = y_test.device(compute_accuracy)(y_test, y_pred)
                    f1 = y_test.device(compute_f1_metric)(y_test, y_pred)
                    fOr = y_test.device(compute_for_metric)(y_test, y_pred)
                    # 揭示（reveal）密态的评估结果
                    acc = sf.reveal(acc)
                    f1 = sf.reveal(f1)
                    fOr = sf.reveal(fOr)
                    # 记录评估指标
                    accs.append(acc)
                    f1s.append(f1)
                    fOrs.append(fOr)
                    # 更新最终评估指标（最后一轮时）
                    if t == n_epochs:
                        if acc > finalacc:
                            finalacc = acc
                        if f1 > finalf1:
                            finalf1 = f1
                        if fOr < finalfOr:
                            finalfOr = fOr
                    print(f"Step {steps}, Accuracy: {acc:.4f}, F1: {f1:.4f}, FOR: {fOr:.4f}")
                steps += 1

        # 训练完成后分发权重到各方
        self.w = self.dispatch_weight()
        print(f"\n📈 最终验证结果:")
        print(f"   • 最终准确率: {finalacc:.4f}")
        print(f"   • 最终F1分数: {finalf1:.4f}")
        print(f"   • 最终误漏率: {finalfOr:.4f}")
        return accs


    def save(self, paths : dict[str, str], ext = 'npy'):
        '''
        保存模型权重到指定路径
        ## Args
        - paths: 保存模型的文件夹路径列表，包含company和partner的路径。例如：
        paths = {
            'company': 'path/to/company/model',
            'partner': 'path/to/partner/model'
        }
        - ext: 保存格式，支持'npy'或'csv'，默认为'npy'
        '''
        # 断言权重必须是联邦数组格式
        assert isinstance(self.w, FedNdarray), "Weights must be a FedNdarray"
        # 分别获取company和partner的权重分区
        w1, w2 = self.w.partitions[self.company], self.w.partitions[self.partner]
        # 构建模型元信息字典
        info = {
            'shape' : (self.in_features, self.out_features),
            'lambda_': float(self.lambda_),
            'approx': bool(self.approx),
            'save_as' : ext
        }
        # 分别在company和partner方执行保存操作
        self.company(lr_save_model)(w1, paths['company'], ext, info)
        self.partner(lr_save_model)(w2, paths['partner'], ext, info)

    @classmethod
    def load(cls, devices, paths):
        """
        从指定路径加载模型权重（类方法）
        ## Args: 
         - devices : 每个字段的值应为SPU或PYU。例如：

           devices = {
            'spu': spu,
            'company': company,
            'partner': partner,
           }

         - paths: 加载模型的文件夹路径列表，包含company和partner的路径。例如：
        paths = {
            'company': 'path/to/company/model/dir',
            'partner': 'path/to/partner/model/dir'
        }
        ## Returns:
        - model: 加载完成的SSLR模型实例
        """
        # 分别在company和partner方执行加载操作
        w1, info1 = devices['company'](lr_load_model, num_returns=2)(paths['company'])
        w2, info2 = devices['partner'](lr_load_model, num_returns=2)(paths['partner'])
        # 揭示（reveal）密态的模型元信息，验证双方的一致性
        info1 = sf.reveal(info1)
        info2 = sf.reveal(info2)
        assert info1 == info2, "Model info mismatch between company and partner"
        info = info1
        # 根据元信息创建模型实例
        model = cls(devices, lambda_=info['lambda_'], approx=info['approx'])
        # 验证设备配置一致
        assert model.company == devices['company'] and model.partner == devices['partner'], "Devices mismatch"
        # 将权重封装为水平分区联邦数组
        model.w = load({model.company: w1, model.partner: w2}, partition_way=PartitionWay.HORIZONTAL)
        # 恢复模型的输入输出维度信息
        model.in_features, model.out_features = info['shape']
        return model
