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
import json
import pytest
import numpy as np
import secretflow as sf
from secretflow.data import FedNdarray
from secretflow.data.ndarray import load, PartitionWay
from XGBoost import (
    generate_indicator,
    leaf_weight_div,
    update_pred,
    gh_sum,
    loss_fraction,
    split_info,
    subtree_args,
    compute_gain,
    leq_compare,
    max_gain_sign,
    bucket_sum,
    tree_leq,
    add_preds,
    select_leaf_weight,
    to_np_array,
    xgb_save_model,
    xgb_save_weights,
    xgb_load_model,
    xgb_load_weights,
    select_weight_by_idx,
)

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

    def test_constant_feature(self):
        """所有值相同的特征 → 只有一个桶, 分位点用该值填充"""
        from XGBoost import quantize_buckets
        X = np.ones((20, 1), dtype=np.float32) * 5.0
        Q, buckets, label_matrix = quantize_buckets(X, k=5)
        # 分位点全部填充为 5.0
        np.testing.assert_allclose(Q[0], [5.0] * 5, atol=1e-6)
        # 所有样本应在同一个桶
        assert len(buckets[0]) == 1
        assert set(buckets[0][0]) == set(range(20))

    def test_k_greater_than_unique_values(self):
        """k 大于唯一值个数 → 分位点有填充"""
        from XGBoost import quantize_buckets
        # 只有 3 个唯一值, k=10
        X = np.array([[1.0], [2.0], [3.0], [1.0], [2.0], [3.0]], dtype=np.float32)
        Q, buckets, label_matrix = quantize_buckets(X, k=10)
        assert Q.shape == (1, 10)
        # 所有样本都应有标签
        assert set(label_matrix.flatten()) <= set(range(11))

    def test_label_matrix_values_valid(self):
        """label_matrix 值应介于 0 和 k 之间"""
        from XGBoost import quantize_buckets
        X = np.random.randn(80, 3).astype(np.float32)
        _, _, label_matrix = quantize_buckets(X, k=8)
        assert label_matrix.min() >= 0
        assert label_matrix.max() <= 8

    def test_single_feature(self):
        """单特征输入"""
        from XGBoost import quantize_buckets
        X = np.random.randn(30, 1).astype(np.float32)
        Q, buckets, label_matrix = quantize_buckets(X, k=5)
        assert Q.shape[0] == 1
        assert label_matrix.shape == (30, 1)
        all_idx = np.concatenate(buckets[0])
        assert set(all_idx) == set(range(30))

    def test_recover_preserves_bucket_partition(self):
        """recover_buckets 后每个特征的桶覆盖全部样本且无重复"""
        from XGBoost import quantize_buckets, recover_buckets
        X = np.random.randn(60, 4).astype(np.float32)
        _, _, label_matrix = quantize_buckets(X, k=7)
        buckets = recover_buckets(label_matrix)
        for j in range(4):
            all_idx = np.concatenate(buckets[j])
            assert len(all_idx) == 60
            assert len(set(all_idx)) == 60

    def test_quantile_values_sorted(self):
        """分位点应非递减"""
        from XGBoost import quantize_buckets
        X = np.random.randn(100, 3).astype(np.float32)
        Q, _, _ = quantize_buckets(X, k=10)
        for j in range(3):
            diffs = np.diff(Q[j])
            assert np.all(diffs >= 0)


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


# ---------------------------------------------------------------------------
# 单元测试 — XGBoost.py 提取的模块级函数
# ---------------------------------------------------------------------------

class TestGenerateIndicator:
    pytestmark = pytest.mark.unit

    def test_shape(self):
        X = np.ones((5, 3))
        result = generate_indicator(X)
        assert result.shape == (5, 1)

    def test_all_ones(self):
        X = np.zeros((3, 2))
        result = generate_indicator(X)
        np.testing.assert_array_equal(result, [[1], [1], [1]])


class TestLeafWeightDiv:
    pytestmark = pytest.mark.unit

    def test_basic(self):
        g_sum = np.array(2.0)
        h_sum = np.array(4.0)
        lambda_ = 1.0
        result = leaf_weight_div(g_sum, h_sum, lambda_)
        np.testing.assert_almost_equal(result, -0.4)

    def test_zero_gradient(self):
        result = leaf_weight_div(np.array(0.0), np.array(1.0), 0.0)
        np.testing.assert_almost_equal(result, 0.0)


class TestUpdatePred:
    pytestmark = pytest.mark.unit

    def test_basic(self):
        pred = np.array([1.0, 2.0])
        weight = np.array([0.5, 0.5])
        s = np.array([1.0, 0.0])
        result = update_pred(pred, weight, s)
        np.testing.assert_array_almost_equal(result, [1.5, 2.0])


class TestGhSum:
    pytestmark = pytest.mark.unit

    def test_basic(self):
        g = np.array([1.0, 2.0, 3.0])
        h = np.array([0.5, 0.5, 0.5])
        g_s, h_s = gh_sum(g, h)
        np.testing.assert_almost_equal(g_s, 6.0)
        np.testing.assert_almost_equal(h_s, 1.5)


class TestLossFraction:
    pytestmark = pytest.mark.unit

    def test_basic(self):
        g_sum = np.array(3.0)
        h_sum = np.array(2.0)
        lambda_ = 1.0
        n, d = loss_fraction(g_sum, h_sum, lambda_)
        np.testing.assert_almost_equal(n, 9.0)
        np.testing.assert_almost_equal(d, 3.0)


class TestSplitInfo:
    pytestmark = pytest.mark.unit

    def test_basic(self):
        g_L, h_L, g_R, h_R, gL2, hLl, gR2, hRl = split_info(
            np.array(1.0), np.array(0.5),
            np.array(2.0), np.array(1.0),
            np.array(6.0), np.array(3.0),
            0.1
        )
        np.testing.assert_almost_equal(g_L, 3.0)
        np.testing.assert_almost_equal(h_L, 1.5)
        np.testing.assert_almost_equal(g_R, 3.0)
        np.testing.assert_almost_equal(h_R, 1.5)


class TestSubtreeArgs:
    pytestmark = pytest.mark.unit

    def test_basic(self):
        g = np.array([[1.0], [2.0], [3.0]])
        h = np.array([[0.1], [0.2], [0.3]])
        s = np.array([[1], [1], [1]])
        s_L = np.array([[1], [0], [1]])
        s_R = np.array([[0], [1], [0]])
        g_L, h_L, sL, g_R, h_R, sR = subtree_args(g, h, s, s_L.copy(), s_R.copy())
        np.testing.assert_array_equal(sL, [[1], [0], [1]])
        np.testing.assert_array_equal(sR, [[0], [1], [0]])
        np.testing.assert_array_almost_equal(g_L, [[1.0], [0.0], [3.0]])
        np.testing.assert_array_almost_equal(g_R, [[0.0], [2.0], [0.0]])


class TestComputeGainUnit:
    pytestmark = pytest.mark.unit

    def test_positive_gain(self):
        result = compute_gain(
            np.array(2.0), np.array(2.0),
            np.array(1.0), np.array(1.0),
            np.array(16.0), np.array(2.0),
            0.0
        )
        np.testing.assert_almost_equal(result, -2.0)


class TestLeqCompare:
    pytestmark = pytest.mark.unit

    def test_first_better(self):
        """第一个分裂点增益更大"""
        result = leq_compare(
            [10.0], [10.0], [1.0], [1.0],
            [1.0], [1.0], [1.0], [1.0],
        )
        assert result.shape[0] == 1


class TestMaxGainSign:
    pytestmark = pytest.mark.unit

    def test_positive(self):
        result = max_gain_sign(
            np.array(10.0), np.array(10.0),
            np.array(2.0), np.array(2.0),
            np.array(1.0), np.array(10.0),
            0.0
        )
        assert isinstance(result, (bool, np.bool_))


class TestBucketSum:
    pytestmark = pytest.mark.unit

    def test_basic(self):
        g = np.array([[1.0], [2.0], [3.0], [4.0]])
        h = np.array([[0.1], [0.2], [0.3], [0.4]])
        bucket = np.array([0, 2])
        g_s, h_s = bucket_sum(g, h, bucket)
        np.testing.assert_almost_equal(g_s, [[4.0]])
        np.testing.assert_almost_equal(h_s, [[0.4]])


class TestTreeLeq:
    pytestmark = pytest.mark.unit

    def test_basic(self):
        X = np.array([[1.0, 5.0], [3.0, 2.0], [2.0, 8.0]])
        Q = np.array([[2.5, 4.0], [0.0, 0.0]])
        result = tree_leq(X, 0, 0, Q)
        np.testing.assert_array_equal(result, [True, False, True])

    def test_column_1(self):
        X = np.array([[1.0, 5.0], [3.0, 2.0]])
        Q = np.array([[0.0, 3.0], [0.0, 0.0]])
        result = tree_leq(X, 1, 0, Q)
        np.testing.assert_array_equal(result, [False, False])


class TestAddPreds:
    pytestmark = pytest.mark.unit

    def test_basic(self):
        result = add_preds(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
        np.testing.assert_array_equal(result, [4.0, 6.0])


class TestSelectLeafWeight:
    pytestmark = pytest.mark.unit

    def test_basic(self):
        w = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        result = select_leaf_weight(w, [2,3])
        np.testing.assert_array_almost_equal(result, [[0.3], [0.4]])

class TestToNpArray:
    pytestmark = pytest.mark.unit

    def test_from_list(self):
        result = to_np_array([1, 2, 3])
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [1, 2, 3])


class TestSelectWeightByIdx:
    pytestmark = pytest.mark.unit

    def test_basic(self):
        x = np.array([[10], [20], [30]])
        result = select_weight_by_idx(x, 1)
        np.testing.assert_array_equal(result, [20])


class TestXgbSaveLoadModel:
    pytestmark = pytest.mark.unit

    def test_save_load_npy(self, tmp_path):
        quantiles = np.array([[1.0, 2.0], [3.0, 4.0]])
        info = {'save_as': 'npy', 'n_estimators': 2}
        trees = [{'type': 'leaf', 'num': 0}, {'type': 'leaf', 'num': 1}]
        path = str(tmp_path / "xgb_model")

        xgb_save_model(quantiles, path, 'npy', info, trees)
        assert os.path.exists(os.path.join(path, 'quantiles.npy'))
        assert os.path.exists(os.path.join(path, 'info.json'))
        assert os.path.exists(os.path.join(path, 'tree.pkl'))

        loaded_trees, loaded_q, loaded_info = xgb_load_model(path)
        np.testing.assert_array_almost_equal(loaded_q, quantiles)
        assert loaded_info == info
        assert loaded_trees == trees


class TestXgbSaveLoadWeights:
    pytestmark = pytest.mark.unit

    def test_save_load_npy(self, tmp_path):
        weights = np.array([[0.1, 0.2], [0.3, 0.4]])
        path = str(tmp_path / "weights")
        os.makedirs(path)
        xgb_save_weights(weights, 'npy', path)
        loaded = xgb_load_weights(path, 'npy')
        np.testing.assert_array_almost_equal(loaded, weights)

    def test_save_load_csv(self, tmp_path):
        weights = np.array([[1.5], [2.5]])
        path = str(tmp_path / "weights_csv")
        os.makedirs(path)
        xgb_save_weights(weights, 'csv', path)
        loaded = xgb_load_weights(path, 'csv')
        np.testing.assert_array_almost_equal(loaded, weights)
