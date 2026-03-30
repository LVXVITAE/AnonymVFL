"""
集成测试 — 推理引擎 InferEngine (对应 test.tex TC5 §5.4.6)

测试内容:
- InferEngine 初始化 (SSLR / SSXGBoost)
- load_data_from_path: 从CSV读取数据
- compute_intersection: 明文交集计算
- infer: SSLR 端到端推理
- score: 评分功能
"""
import os
import tempfile
import pytest
import numpy as np
import pandas as pd
import secretflow as sf
from secretflow.data.ndarray import load, PartitionWay
from infer import read_dataset, filter_data, compute_common_keys, make_prediction_df

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_csv_pair(tmpdir, n_samples=50, n_company_features=8, n_partner_features=9, overlap_ratio=0.8):
    """创建一对 CSV 文件模拟 company 和 partner 的数据.

    Company CSV: id, feat1..featN, Revenue
    Partner CSV: id, feat1..featN (无标签)
    """
    rng = np.random.RandomState(42)

    n_overlap = int(n_samples * overlap_ratio)
    company_ids = list(range(1, n_samples + 1))
    partner_ids = list(range(n_samples - n_overlap + 1, 2 * n_samples - n_overlap + 1))

    # Company data
    company_feats = rng.randn(n_samples, n_company_features).astype(np.float32)
    company_labels = rng.randint(0, 2, size=n_samples)
    company_df = pd.DataFrame(company_feats, columns=[f"f{i}" for i in range(n_company_features)])
    company_df.insert(0, "id", company_ids)
    company_df["Revenue"] = company_labels

    # Partner data
    partner_feats = rng.randn(n_samples, n_partner_features).astype(np.float32)
    partner_df = pd.DataFrame(partner_feats, columns=[f"g{i}" for i in range(n_partner_features)])
    partner_df.insert(0, "id", partner_ids)

    company_path = os.path.join(tmpdir, "company_data.csv")
    partner_path = os.path.join(tmpdir, "partner_data.csv")
    company_df.to_csv(company_path, index=False)
    partner_df.to_csv(partner_path, index=False)

    expected_overlap = set(company_ids) & set(partner_ids)
    return company_path, partner_path, expected_overlap


def _train_and_save_sslr(devices, tmpdir, make_binary_data):
    """训练一个小型 SSLR 并保存, 返回 model paths 和 split_col."""
    from LR import SSLR

    company = devices["company"]
    partner = devices["partner"]
    spu = devices["spu"]

    n_features = 8 + 9  # 模拟 company(8) + partner(9)
    X, y = make_binary_data(n_samples=100, n_features=n_features, seed=7)
    split_col = 8

    train_X = sf.to(company, X.astype(np.float32)).to(spu)
    train_y = sf.to(company, y.astype(np.float32)).to(spu)

    test_X = load(
        {company: sf.to(company, X[:, :split_col].astype(np.float32)),
         partner: sf.to(partner, X[:, split_col:].astype(np.float32))},
        partition_way=PartitionWay.VERTICAL,
    )
    test_y = sf.to(company, y.astype(np.float32))

    model = SSLR(devices, approx=True)
    model.fit(train_X, train_y, X_test=test_X, y_test=test_y,
              n_epochs=2, batch_size=32, val_steps=1, lr=0.1)

    paths = {
        "company": os.path.join(tmpdir, "sslr_company"),
        "partner": os.path.join(tmpdir, "sslr_partner"),
    }
    model.save(paths, ext="npy")
    return paths


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestInferEngineInit:

    def test_init_sslr(self, devices):
        from infer import InferEngine
        engine = InferEngine(devices, model="SSLR")
        assert engine.model_type is not None

    def test_init_ssxgboost(self, devices):
        from infer import InferEngine
        engine = InferEngine(devices, model="SSXGBoost")
        assert engine.model_type is not None

    def test_init_invalid_model(self, devices):
        from infer import InferEngine
        with pytest.raises(ValueError, match="Unsupported model type"):
            InferEngine(devices, model="InvalidModel")


class TestInferEngineDataLoading:

    def test_load_data_from_path(self, devices):
        """验证 load_data_from_path 返回 4 个 PYUObject"""
        from infer import InferEngine

        with tempfile.TemporaryDirectory() as tmpdir:
            company_path, partner_path, _ = _create_csv_pair(tmpdir)
            engine = InferEngine(devices, model="SSLR")
            ck, cd, pk, pd_ = engine.load_data_from_path({
                "company": company_path,
                "partner": partner_path,
            })
            # 应该是 PYUObject
            assert ck is not None
            assert cd is not None

    def test_compute_intersection(self, devices):
        """验证交集计算的正确性"""
        from infer import InferEngine

        with tempfile.TemporaryDirectory() as tmpdir:
            company_path, partner_path, expected_overlap = _create_csv_pair(tmpdir, n_samples=30)
            engine = InferEngine(devices, model="SSLR")
            ck, cd, pk, pd_ = engine.load_data_from_path({
                "company": company_path,
                "partner": partner_path,
            })
            keys, X = engine.compute_intersection(ck, cd, pk, pd_)
            # 交集键数与预期一致
            assert len(keys) == len(expected_overlap)
            # 所有交集键应在预期内
            for k in keys:
                assert int(k) in expected_overlap or str(k) in {str(x) for x in expected_overlap}


class TestInferEngineEndToEnd:

    def test_sslr_infer_pipeline(self, devices, make_binary_data):
        """SSLR 端到端推理: 训练→保存→加载→推理→评分"""
        from infer import InferEngine

        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. 训练并保存
            model_paths = _train_and_save_sslr(devices, tmpdir, make_binary_data)

            # 2. 创建推理数据
            company_path, partner_path, _ = _create_csv_pair(
                tmpdir, n_samples=40, n_company_features=8, n_partner_features=9
            )

            # 3. 推理
            engine = InferEngine(devices, model="SSLR")
            engine.load_model(model_paths)

            ck, cd, pk, pd_ = engine.load_data_from_path({
                "company": company_path,
                "partner": partner_path,
            })
            keys, X = engine.compute_intersection(ck, cd, pk, pd_)
            pred = engine.infer(device=devices["company"])
            result = sf.reveal(pred)

            # 结果应是 DataFrame
            assert isinstance(result, pd.DataFrame)
            assert "id" in result.columns
            assert "prediction" in result.columns
            assert len(result) == len(keys)


# ---------------------------------------------------------------------------
# 单元测试 — infer.py 提取的模块级函数
# ---------------------------------------------------------------------------

class TestReadDataset:
    pytestmark = pytest.mark.unit

    def test_basic_read(self, tmp_path):
        """读取CSV文件返回keys列表和DataFrame"""
        csv_path = tmp_path / "data.csv"
        df = pd.DataFrame({'id': [1, 2, 3], 'feat1': [0.1, 0.2, 0.3], 'feat2': [0.4, 0.5, 0.6]})
        df.to_csv(csv_path, index=False)
        keys, data = read_dataset(str(csv_path))
        assert keys == ['1', '2', '3']
        assert isinstance(data, pd.DataFrame)
        assert data.shape == (3, 3)

    def test_string_keys(self, tmp_path):
        """keys应转换为字符串"""
        csv_path = tmp_path / "data.csv"
        df = pd.DataFrame({'id': ['alice', 'bob'], 'x': [1.0, 2.0]})
        df.to_csv(csv_path, index=False)
        keys, _ = read_dataset(str(csv_path))
        assert keys == ['alice', 'bob']

    def test_empty_dataset(self, tmp_path):
        """空数据集返回空keys"""
        csv_path = tmp_path / "empty.csv"
        df = pd.DataFrame({'id': pd.Series([], dtype=str), 'x': pd.Series([], dtype=float)})
        df.to_csv(csv_path, index=False)
        keys, data = read_dataset(str(csv_path))
        assert keys == []
        assert len(data) == 0


class TestFilterData:
    pytestmark = pytest.mark.unit

    def test_basic_filter(self):
        """按keys过滤数据并返回numpy数组"""
        df = pd.DataFrame({'id': ['a', 'b', 'c', 'd'], 'f1': [1, 2, 3, 4], 'f2': [5, 6, 7, 8]})
        result = filter_data(df, ['b', 'd'])
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 2)

    def test_removes_revenue_column(self):
        """如有Revenue列则去除"""
        df = pd.DataFrame({'id': ['a', 'b'], 'f1': [1, 2], 'Revenue': [100, 200]})
        result = filter_data(df, ['a', 'b'])
        assert result.shape == (2, 1)

    def test_no_revenue_column(self):
        """无Revenue列时保留所有特征"""
        df = pd.DataFrame({'id': ['a', 'b'], 'f1': [1, 2], 'f2': [3, 4]})
        result = filter_data(df, ['a', 'b'])
        assert result.shape == (2, 2)

    def test_sorted_by_id(self):
        """结果按id排序"""
        df = pd.DataFrame({'id': ['c', 'a', 'b'], 'f1': [3, 1, 2]})
        result = filter_data(df, ['a', 'b', 'c'])
        np.testing.assert_array_equal(result.flatten(), [1, 2, 3])

    def test_filter_no_match(self):
        """无匹配项返回空数组"""
        df = pd.DataFrame({'id': ['a', 'b'], 'f1': [1, 2]})
        result = filter_data(df, ['x', 'y'])
        assert result.shape[0] == 0


class TestComputeCommonKeys:
    pytestmark = pytest.mark.unit

    def test_basic_intersection(self):
        result = compute_common_keys(['a', 'b', 'c'], ['b', 'c', 'd'])
        assert set(result) == {'b', 'c'}

    def test_no_overlap(self):
        result = compute_common_keys(['a', 'b'], ['c', 'd'])
        assert result == []

    def test_full_overlap(self):
        result = compute_common_keys(['a', 'b'], ['a', 'b'])
        assert set(result) == {'a', 'b'}

    def test_empty_inputs(self):
        assert compute_common_keys([], ['a']) == []
        assert compute_common_keys(['a'], []) == []


class TestMakePredictionDf:
    pytestmark = pytest.mark.unit

    def test_basic(self):
        keys = ['a', 'b', 'c']
        y = np.array([0.1, 0.9, 0.5])
        result = make_prediction_df(keys, y)
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ['id', 'prediction']
        assert list(result['id']) == keys
        np.testing.assert_array_almost_equal(result['prediction'].values, y)

    def test_2d_predictions(self):
        keys = ['x', 'y']
        y = np.array([[0.3], [0.7]])
        result = make_prediction_df(keys, y)
        assert result.shape == (2, 2)
