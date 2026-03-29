import numpy as np
import jax
import jax.numpy as jnp
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import secretflow as sf
from secretflow.device import PYUObject, SPUObject
from secretflow.data import FedNdarray
from secretflow import SPU, PYU
out_dom = int(2**16)

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def approx_sigmoid(x: jnp.ndarray):
    """
    Compute approximated sigmoid using piecewise function
    """

    return jnp.clip(x + 1/2, 0, 1)


def sigmoid(x: jnp.ndarray) -> jnp.ndarray:
    """
    Computes the sigmoid function.
    """
    return 1.0 / (1.0 + jnp.exp(-x))


def softmax(x: jnp.ndarray) -> jnp.ndarray:
    """
    Computes the softmax function.
    """
    e_x = jnp.exp(x - jnp.max(x, axis=-1, keepdims=True))
    return e_x / jnp.sum(e_x, axis=-1, keepdims=True)


def cross_entropy(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    """
    Computes the cross-entropy loss.
    """
    return -jnp.sum(y_true * jnp.log(y_pred + 1e-12))


class ApproxSigmoidCrossEntropy:
    @staticmethod
    def loss(y_true: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the sigmoid cross-entropy loss.
        """
        y_pred = approx_sigmoid(z)
        return cross_entropy(y_true, y_pred)

    @staticmethod
    def grad(y_true: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the gradient of the sigmoid cross-entropy loss.
        """
        y_pred = approx_sigmoid(z)
        return y_pred - y_true

    @staticmethod
    def hess(y_true: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the hessian of the sigmoid cross-entropy loss.
        """
        y_pred = approx_sigmoid(z)
        return y_pred * (1.0 - y_pred)


class SigmoidCrossEntropy:
    @staticmethod
    def loss(y_true: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the sigmoid cross-entropy loss.
        """
        y_pred = sigmoid(z)
        return cross_entropy(y_true, y_pred)

    @staticmethod
    def grad(y_true: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the gradient of the sigmoid cross-entropy loss.
        """
        y_pred = sigmoid(z)
        return y_pred - y_true

    @staticmethod
    def hess(y_true: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the hessian of the sigmoid cross-entropy loss.
        """
        y_pred = sigmoid(z)
        return y_pred * (1.0 - y_pred)


class SoftmaxCrossEntropy:
    @staticmethod
    def loss(y_true: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the softmax cross-entropy loss.
        """
        y_pred = softmax(z)
        return cross_entropy(y_true, y_pred)

    @staticmethod
    def grad(y_true: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the gradient of the softmax cross-entropy loss.
        """
        y_pred = softmax(z)
        return y_pred - y_true

    @staticmethod
    def hess(y_true: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the hessian of the softmax cross-entropy loss.
        """
        y_pred = softmax(z)
        return y_pred * (1.0 - y_pred)


def mean_square_error(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    """
    Computes the mean square error loss.
    """
    return jnp.mean((y_true - y_pred) ** 2)


class MeanSquare:
    @staticmethod
    def loss(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the mean square loss.
        """
        return mean_square_error(y_true, y_pred)

    @staticmethod
    def grad(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the gradient of the mean square loss.
        """
        return 2 * (y_pred - y_true) / y_true.shape[0]

    @staticmethod
    def hess(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the hessian of the mean square loss.
        """
        return 2 / y_true.shape[0]


def to_int_labels(logits: np.ndarray):
    # 将logit转化为整数标签
    if logits.shape[1] == 1:
        return np.round(logits)
    else:
        return np.argmax(logits, axis=1)


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray):
    y_true = y_true.reshape(-1, 1)
    y_pred = y_pred.reshape(-1, 1)
    return np.mean(y_true == y_pred)


def compute_f1_metric(y_true: np.ndarray, y_pred: np.ndarray):
    y_true = y_true.reshape(-1, 1)
    y_pred = y_pred.reshape(-1, 1)
    positive_mask = (y_true == 0)
    negative_mask = (y_true == 1)
    pred_positive = (y_pred == 0)
    pred_negative = (y_pred == 1)
    tp = np.sum(positive_mask & pred_positive)
    fp = np.sum(negative_mask & pred_positive)
    fn = np.sum(positive_mask & pred_negative)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    return 2 * (precision * recall) / (precision + recall + 1e-8)


def compute_for_metric(y_true: np.ndarray, y_pred: np.ndarray):
    y_true = y_true.reshape(-1, 1)
    y_pred = y_pred.reshape(-1, 1)
    positive_mask = (y_true == 0)
    negative_mask = (y_true == 1)
    pred_negative = (y_pred == 1)
    fn = np.sum(positive_mask & pred_negative)
    tn = np.sum(negative_mask & pred_negative)
    return fn / (fn + tn + 1e-8)

# 加载数据集，开发测试用


def load_dataset(dataset: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    实现了训练测试集划分以及标签（y）01/独热编码
    """
    if dataset == "pima" or dataset == "lbw" or dataset == "pcs" or dataset == "uis":
        data = pd.read_csv(os.path.join(
            "Datasets", f"{dataset}.csv")).to_numpy()
        train_data, test_data = train_test_split(data, shuffle=False)
        train_X = train_data[:, :-1]
        train_y = train_data[:, -1].reshape(-1, 1)
        test_X = test_data[:, :-1]
        test_y = test_data[:, -1].reshape(-1, 1)

    elif dataset == "gisette" or dataset == "arcene":
        folder = os.path.join("Datasets", dataset)
        train_X = np.loadtxt(os.path.join(folder, f"{dataset}_train.data"))
        train_y = np.loadtxt(os.path.join(folder, f"{dataset}_train.labels"))
        train_y[train_y == -1] = 0
        train_y = train_y.reshape(-1, 1)
        test_X = np.loadtxt(os.path.join(folder, f"{dataset}_valid.data"))
        test_y = np.loadtxt(os.path.join(folder, f"{dataset}_valid.labels"))
        test_y[test_y == -1] = 0
        test_y = test_y.reshape(-1, 1)

    elif dataset == "mnist":
        from sklearn.datasets import fetch_openml
        mnist = fetch_openml('mnist_784', version=1, as_frame=False)
        X = mnist.data
        y = mnist.target
        train_X, test_X, train_y, test_y = train_test_split(
            X, y, shuffle=False)
        train_y = train_y.astype(int).reshape(-1, 1)
        from sklearn.preprocessing import OneHotEncoder
        train_y = OneHotEncoder().fit_transform(train_y).toarray()
        test_y = test_y.astype(int).reshape(-1, 1)

    elif dataset == "risk":
        dir_path = os.path.join("Datasets", "data", "data")
        train = pd.read_csv(os.path.join(dir_path, "risk_assessment_all.csv"))
        test = pd.read_csv(os.path.join(
            dir_path, "risk_assessment_all_test.csv"))
        train_X = train.drop(columns=["id", "y"]).to_numpy()
        train_y = train["y"].to_numpy().reshape(-1, 1)
        test_X = test.drop(columns=["id", "y"]).to_numpy()
        test_y = test["y"].to_numpy().reshape(-1, 1)

    elif dataset == "breast":
        dir_path = os.path.join("Datasets", "data", "data")
        guest = pd.read_csv(os.path.join(dir_path, "breast_hetero_guest.csv"))
        host = pd.read_csv(os.path.join(dir_path, "breast_hetero_host.csv"))
        all = pd.concat([host, guest], join='inner', axis=1)
        X = all.drop(columns=["id", "y"]).to_numpy()
        y = all["y"].to_numpy().reshape(-1, 1)
        train_X, test_X, train_y, test_y = train_test_split(X, y, shuffle=True)

    elif dataset == "shop":
        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        company_train_csv = os.path.join(project_dir, "company", "host_train.csv")
        company_test_csv = os.path.join(project_dir, "company", "host_test.csv")
        partner_train_csv = os.path.join(project_dir, "partner", "guest_train.csv")
        partner_test_csv = os.path.join(project_dir, "partner", "guest_test.csv")
        company_train = pd.read_csv(company_train_csv)
        company_test = pd.read_csv(company_test_csv)
        partner_train = pd.read_csv(partner_train_csv)
        partner_test = pd.read_csv(partner_test_csv)

        # 取交集 (by id)
        train_merged = pd.merge(company_train, partner_train, on="id", how="inner").sort_values("id")
        test_merged = pd.merge(company_test, partner_test, on="id", how="inner").sort_values("id")

        # Company features: columns between id and Revenue (exclusive)
        company_feat_cols = [c for c in company_train.columns if c not in ("id", "Revenue")]
        partner_feat_cols = [c for c in partner_train.columns if c != "id"]
        split_col = len(company_feat_cols)

        train_X = train_merged[company_feat_cols + partner_feat_cols].to_numpy(dtype=np.float32)
        train_y = train_merged["Revenue"].to_numpy(dtype=np.float32).reshape(-1, 1)
        test_X = test_merged[company_feat_cols + partner_feat_cols].to_numpy(dtype=np.float32)
        test_y = test_merged["Revenue"].to_numpy(dtype=np.float32).reshape(-1, 1)

    return train_X, train_y, test_X, test_y


def Singleton(cls):  # 这是一个函数，目的是要实现一个“装饰器”，而且是对类型的装饰器
    '''
    cls:表示一个类名，即所要设计的单例类名称，
        因为python一切皆对象，故而类名同样可以作为参数传递
    '''
    instance = {}

    def singleton(*args, **kargs):
        if cls not in instance:
            # 如果没有cls这个类，则创建，并且将这个cls所创建的实例，保存在一个字典中
            instance[cls] = cls(*args, **kargs)
        return instance[cls]

    return singleton


@Singleton
class MPCInitializer:
    """
    初始化secretflow的SPU和PYU环境
    这里添加一个装饰器以实现单例模式
    目前仅支持仿真模式
    根据secretflow的文档，仅修改此处的初始化方式而几乎无需改动其他源码即可实现分布式部署
    """

    def __init__(self, mode='single_sim', ray_head_addr="", cluster_def={}):
        self.mode = mode
        if mode == 'single_sim':
            sf.init(['company', 'partner', 'coordinator'],
                    address='local',
                    )
            self.config = sf.utils.testing.cluster_def(
                parties=['company', 'partner', 'coordinator'])
        if mode == 'multi_sim':
            sf.init(['company', 'partner', 'coordinator'],
                    address=ray_head_addr,
                    )
            self.config = cluster_def
        if mode == 'multi_distributed':
            # 分布式模式，每个节点只初始化自己需要的角色
            sf.init(['company', 'partner', 'coordinator'],
                    address=ray_head_addr,
                    )
            self.config = cluster_def
        self.spu = sf.SPU(self.config)
        self.company, self.partner, self.coordinator = sf.PYU(
            'company'), sf.PYU('partner'), sf.PYU('coordinator')
        encoding = {
            'cleartext_type': 'DT_F32',
            'encoder': 'FloatEncoder'
        }
        heu_config = sf.utils.testing.heu_config(
            sk_keeper='company', evaluators=['partner'])
        heu_config['encoding'] = encoding
        self.company_heu = sf.HEU(
            heu_config, self.spu.cluster_def['runtime_config']['field'])

        heu_config = sf.utils.testing.heu_config(
            sk_keeper='partner', evaluators=['company'])
        heu_config['encoding'] = encoding
        self.partner_heu = sf.HEU(
            heu_config, self.spu.cluster_def['runtime_config']['field'])


class SSML:
    def __init__(self, devices: dict):
        """
        初始化秘密共享机器学习模型基类
        ## Args:
         - devices : 每个字段的值应为SPU或PYU。例如：

           devices = {
            'spu': spu,
            'company': company,
            'partner': partner,
           }
        """
        assert 'company' in devices and 'partner' in devices and isinstance(
            devices['company'], PYU) and isinstance(devices['partner'], PYU), \
            "devices must contain 'company' and 'partner' as PYU devices"
        self.company = devices['company']
        self.partner = devices['partner']
        self.devices = devices
        if 'spu' in devices and isinstance(devices['spu'], SPU):
            self.spu = devices['spu']
        else:
            self.spu = None

    def fit(self, X: SPUObject, y, X_test: FedNdarray = None, y_test: PYUObject = None):
        raise NotImplementedError("Subclasses should implement this method.")

    def predict(self, X: FedNdarray, device: PYU) -> PYUObject:
        raise NotImplementedError("Subclasses should implement this method.")

    @staticmethod
    def score(y_true: PYUObject, y_pred: PYUObject) -> dict:
        assert y_true.device == y_pred.device, "y_true and y_pred must be on the same device"
        val_device = y_true.device
        acc_score = val_device(compute_accuracy)(y_true, y_pred)
        f1_score = val_device(compute_f1_metric)(y_true, y_pred)
        for_score = val_device(compute_for_metric)(y_true, y_pred)
        return {
            'accuracy': sf.reveal(acc_score),
            'f1_metric': sf.reveal(f1_score),
            'for_metric': sf.reveal(for_score)
        }

    def save(self, paths: dict, ext: str = 'npy'):
        raise NotImplementedError("Subclasses should implement this method.")

    @classmethod
    def load(cls, devices: dict, paths: dict):
        raise NotImplementedError("Subclasses should implement this method.")
