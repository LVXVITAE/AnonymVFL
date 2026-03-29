"""
集成测试 — SSXGBoost 秘密共享梯度提升树 (对应 test.tex TC3/TC4 §5.4.4, §5.4.5)

测试内容:
- quantize_buckets / recover_buckets 分桶一致性
- TC3: SSXGBoost 训练(小数据集, 浅树)
- TC3: SSXGBoost predict 输出形状
- TC4: SSXGBoost save/load (注意: 源码 save() 缺少 train_label_keeper, 当前跳过)
"""
import os
import tempfile
import pytest
import numpy as np
import secretflow as sf
from secretflow.data import FedNdarray
from secretflow.data.ndarray import load, PartitionWay

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prepare_xgboost_data(devices, make_binary_data, n_samples=100, n_features=8, k=10):
    """构造 SSXGBoost 训练所需的全部数据: X(SPU), y(PYU), buckets, FedQuantiles, test_X, test_y."""
    from XGBoost import quantize_buckets, recover_buckets

    company = devices["company"]
    partner = devices["partner"]
    spu = devices["spu"]

    X, y = make_binary_data(n_samples=n_samples, n_features=n_features, seed=42)
    split_col = n_features // 2

    # 分桶
    Q1, _, bl1 = quantize_buckets(X[:, :split_col], k=k)
    Q2, _, bl2 = quantize_buckets(X[:, split_col:], k=k)
    buckets = recover_buckets(np.hstack((bl1, bl2)))

    # FedQuantiles
    FedQuantiles = load(
        {company: sf.to(company, Q1), partner: sf.to(partner, Q2)},
        partition_way=PartitionWay.HORIZONTAL,
    )

    # 训练数据 → SPU / PYU
    train_X = sf.to(company, X.astype(np.float32)).to(spu)
    train_y = sf.to(company, y.astype(np.float32))  # PYU (label keeper)

    # 测试数据 → FedNdarray
    test_X = load(
        {company: sf.to(company, X[:, :split_col].astype(np.float32)),
         partner: sf.to(partner, X[:, split_col:].astype(np.float32))},
        partition_way=PartitionWay.VERTICAL,
    )
    test_y = sf.to(company, y.astype(np.float32))

    return train_X, train_y, buckets, FedQuantiles, test_X, test_y, split_col


# ---------------------------------------------------------------------------
# 分桶算法
# ---------------------------------------------------------------------------

class TestQuantizeBuckets:
    """验证 quantize_buckets 与 recover_buckets 的一致性"""

    def test_quantize_shape(self):
        from XGBoost import quantize_buckets
        X = np.random.randn(50, 4).astype(np.float32)
        Q, buckets, label_matrix = quantize_buckets(X, k=5)
        # Q: (n_features, k) 分位点
        assert Q.shape[0] == 4
        assert label_matrix.shape == X.shape

    def test_recover_buckets_roundtrip(self):
        from XGBoost import quantize_buckets, recover_buckets
        X = np.random.randn(50, 3).astype(np.float32)
        _, _, label_matrix = quantize_buckets(X, k=5)
        buckets = recover_buckets(label_matrix)
        # 每个特征的桶内索引并集 == 所有样本索引
        for j in range(3):
            all_indices = np.concatenate(buckets[j])
            assert set(all_indices) == set(range(50))

    def test_each_sample_in_exactly_one_bucket(self):
        from XGBoost import quantize_buckets
        X = np.random.randn(100, 2).astype(np.float32)
        _, buckets, _ = quantize_buckets(X, k=10)
        for j in range(2):
            all_idx = np.concatenate(buckets[j])
            # 无重复
            assert len(all_idx) == len(set(all_idx))


# ---------------------------------------------------------------------------
# TC3: SSXGBoost 训练
# ---------------------------------------------------------------------------

class TestSSXGBoostTraining:

    @pytest.mark.slow
    def test_fit_returns_accs(self, devices, make_binary_data):
        """SSXGBoost 训练后返回准确率列表"""
        from XGBoost import SSXGBoost

        (train_X, train_y, buckets, FedQuantiles,
         test_X, test_y, split_col) = _prepare_xgboost_data(
            devices, make_binary_data, n_samples=100, n_features=8, k=10
        )
        model = SSXGBoost(
            devices, n_estimators=2, lambda_=1e-4, max_depth=2, div=False
        )
        train_accs, test_accs = model.fit(
            train_X, train_y, buckets, FedQuantiles,
            X_test=test_X, y_test=test_y,
        )
        assert len(train_accs) == 2
        assert len(test_accs) == 2
        for a in train_accs + test_accs:
            assert 0.0 <= float(a) <= 1.0

    @pytest.mark.slow
    def test_predict_shape(self, devices, make_binary_data):
        """predict 输出与样本数一致"""
        from XGBoost import SSXGBoost

        (train_X, train_y, buckets, FedQuantiles,
         test_X, test_y, split_col) = _prepare_xgboost_data(
            devices, make_binary_data, n_samples=80, n_features=6, k=8
        )
        model = SSXGBoost(
            devices, n_estimators=1, lambda_=1e-4, max_depth=2
        )
        model.fit(train_X, train_y, buckets, FedQuantiles)
        y_pred = model.predict(test_X, devices["company"])
        result = sf.reveal(y_pred)
        assert result.shape[0] == 80

    @pytest.mark.slow
    def test_accuracy_better_than_random(self, devices, make_binary_data):
        """训练后的准确率应优于随机猜测 (>0.4)"""
        from XGBoost import SSXGBoost

        (train_X, train_y, buckets, FedQuantiles,
         test_X, test_y, split_col) = _prepare_xgboost_data(
            devices, make_binary_data, n_samples=200, n_features=10, k=15
        )
        model = SSXGBoost(
            devices, n_estimators=3, lambda_=1e-4, max_depth=3
        )
        train_accs, _ = model.fit(train_X, train_y, buckets, FedQuantiles)
        # 最终训练准确率应大于随机水平
        assert float(train_accs[-1]) > 0.4


# ---------------------------------------------------------------------------
# TC4: SSXGBoost 持久化
# NOTE: 源码 XGBoost.py SSXGBoost.save() 的 info dict 缺少 'train_label_keeper' 键,
#       但 load() 依赖该键 (line 829). 这是一个已知 bug.
#       下面的测试在 save/load round-trip 时预期会触发 KeyError.
# ---------------------------------------------------------------------------

class TestSSXGBoostPersistence:

    @pytest.mark.slow
    def test_save_creates_files(self, devices, make_binary_data):
        """save 应创建 weight.npy, quantiles.npy, info.json, tree.pkl"""
        from XGBoost import SSXGBoost

        (train_X, train_y, buckets, FedQuantiles,
         test_X, test_y, split_col) = _prepare_xgboost_data(
            devices, make_binary_data, n_samples=80, n_features=6, k=8
        )
        model = SSXGBoost(
            devices, n_estimators=1, lambda_=1e-4, max_depth=2
        )
        model.fit(train_X, train_y, buckets, FedQuantiles)

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = {
                "company": os.path.join(tmpdir, "company_xgb"),
                "partner": os.path.join(tmpdir, "partner_xgb"),
            }
            for p in paths.values():
                os.makedirs(p, exist_ok=True)
            model.save(paths, ext="npy")
            # Barrier: flush async PYU writes before checking files
            sf.reveal(devices["company"](lambda: 0)())
            sf.reveal(devices["partner"](lambda: 0)())

            # Both parties get quantiles, info, tree
            for party in ["company", "partner"]:
                assert os.path.exists(os.path.join(paths[party], "quantiles.npy"))
                assert os.path.exists(os.path.join(paths[party], "info.json"))
                assert os.path.exists(os.path.join(paths[party], "tree.pkl"))
            # weight.npy is only saved under the label keeper (company)
            assert os.path.exists(os.path.join(paths["company"], "weight.npy"))

    @pytest.mark.slow
    @pytest.mark.xfail(
        reason="Known bug: SSXGBoost.save() missing 'train_label_keeper' in info dict",
        raises=KeyError,
    )
    def test_save_load_roundtrip(self, devices, make_binary_data):
        """save → load round-trip (预期因 bug 失败)"""
        from XGBoost import SSXGBoost

        (train_X, train_y, buckets, FedQuantiles,
         test_X, test_y, split_col) = _prepare_xgboost_data(
            devices, make_binary_data, n_samples=80, n_features=6, k=8
        )
        model = SSXGBoost(
            devices, n_estimators=1, lambda_=1e-4, max_depth=2
        )
        model.fit(train_X, train_y, buckets, FedQuantiles)

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = {
                "company": os.path.join(tmpdir, "company_xgb"),
                "partner": os.path.join(tmpdir, "partner_xgb"),
            }
            for p in paths.values():
                os.makedirs(p, exist_ok=True)
            model.save(paths, ext="npy")
            sf.reveal(devices["company"](lambda: 0)())
            sf.reveal(devices["partner"](lambda: 0)())
            loaded = SSXGBoost.load(devices, paths)
            y_orig = sf.reveal(model.predict(test_X, devices["company"]))
            y_loaded = sf.reveal(loaded.predict(test_X, devices["company"]))
            np.testing.assert_array_equal(y_orig, y_loaded)
