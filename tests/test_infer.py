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
