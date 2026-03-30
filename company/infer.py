from secretflow import SPU, PYU
from secretflow.data.ndarray import load, PartitionWay
import secretflow as sf
import pandas as pd
from LR import SSLR
from XGBoost import SSXGBoost
from common import SSML


def read_dataset(path: str):
    """读取CSV数据集，返回（keys列表, DataFrame）"""
    data = pd.read_csv(path)
    keys = data.iloc[:, 0].astype(str).tolist()
    return keys, data


def filter_data(data: pd.DataFrame, keys: list):
    """按keys过滤数据，去掉id列和Revenue列（如存在），返回numpy数组"""
    data = data[data.iloc[:, 0].astype(str).isin(keys)]
    data = data.sort_values(by=data.columns[0])
    feature_df = data.iloc[:, 1:]
    if 'Revenue' in feature_df.columns:
        feature_df = feature_df.drop(columns=['Revenue'])
    return feature_df.to_numpy()


def compute_common_keys(x, y):
    """计算两个列表的交集"""
    return list(set(x) & set(y))


def make_prediction_df(keys, y):
    """将keys和预测值组合成DataFrame"""
    return pd.DataFrame({
        'id': keys,
        'prediction': y.flatten()
    })


class InferEngine:
    def __init__(self, devices: dict, model: str):
        """
        初始化推理引擎，可不初始化SPU，但必须初始化coordinator和各参与方PYU
        ## Args:
         - devices : 每个字段的值应为SPU或PYU。例如：

           devices = {
            'company': company,
            'partner': partner,
            'coordinator': coordinator,
           }
        """
        assert 'company' in devices and 'partner' in devices and isinstance(
            devices['company'], PYU) and isinstance(devices['partner'], PYU), "devices must contain 'company' and 'partner' as PYU devices"
        self.company = devices['company']
        self.partner = devices['partner']
        self.coordinator = devices['coordinator']
        if 'spu' in devices and isinstance(devices['spu'], SPU):
            self.spu = devices['spu']
        else:
            self.spu = None
        if model == 'SSLR':
            self.model_type = SSLR
        elif model == 'SSXGBoost':
            self.model_type = SSXGBoost
        else:
            raise ValueError(f"Unsupported model type: {model}")

    def load_model(self, paths: dict):
        self.model = self.model_type.load(
            {'company': self.company, 'partner': self.partner}, paths)
        return self.model

    def load_data_from_path(self, paths: dict):
        """
        加载推理数据，并对推理数据求交集
        ## Args:
         - paths : 每个字段的值应为数据csv文件路径。例如：

           paths = {
            'company': 'path/to/company/data.csv',
            'partner': 'path/to/partner/data.csv',
           }
        """

        company_keys, company_data = self.company(
            read_dataset, num_returns=2)(paths['company'])
        partner_keys, partner_data = self.partner(
            read_dataset, num_returns=2)(paths['partner'])
        return company_keys, company_data, partner_keys, partner_data

    def compute_intersection(self, company_keys, company_data, partner_keys, partner_data):
        company_keys = company_keys.to(self.coordinator)
        partner_keys = partner_keys.to(self.coordinator)
        common_keys = self.coordinator(
            compute_common_keys
        )(company_keys, partner_keys)
        self.keys = sf.reveal(common_keys)

        company_data = self.company(filter_data)(company_data, self.keys)
        partner_data = self.partner(filter_data)(partner_data, self.keys)
        self.X = load(
            {
                self.company: company_data,
                self.partner: partner_data
            },
            partition_way=PartitionWay.VERTICAL
        )
        return self.keys, self.X

    def infer(self, device: PYU):
        pred_y = self.model.predict(self.X, device=device)
        pred_y = device(make_prediction_df)(self.keys, pred_y)
        return pred_y

    @staticmethod
    def score(y_true, y_pred):
        return SSML.score(y_true, y_pred)
