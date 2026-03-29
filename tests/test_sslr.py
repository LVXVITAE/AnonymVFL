"""
集成测试 — SSLR 秘密共享逻辑回归 (对应 test.tex TC2/TC4 §5.4.3, §5.4.5)

测试内容:
- TC2: SSLR 训练收敛性 (binary, approx/non-approx)
- TC4: SSLR 模型持久化 (save/load round-trip, 权重一致性)
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

def _prepare_sslr_data(devices, make_binary_data, n_samples=200, n_features=10):
    """构造 SSLR 训练/测试数据并上传到 SPU/PYU 设备."""
    company = devices["company"]
    partner = devices["partner"]
    spu = devices["spu"]

    X, y = make_binary_data(n_samples=n_samples, n_features=n_features, seed=42)
    split_col = n_features // 2

    X_company = X[:, :split_col]
    X_partner = X[:, split_col:]

    # 训练数据 → SPU
    train_X = sf.to(company, X).to(spu)
    train_y = sf.to(company, y).to(spu)

    # 测试数据 → FedNdarray (纵向划分)
    test_X = load(
        {company: sf.to(company, X_company), partner: sf.to(partner, X_partner)},
        partition_way=PartitionWay.VERTICAL,
    )
    test_y = sf.to(company, y)

    return train_X, train_y, test_X, test_y, split_col


# ---------------------------------------------------------------------------
# TC2: SSLR 训练
# ---------------------------------------------------------------------------

class TestSSLRTraining:

    def test_fit_approx_returns_accs(self, devices, make_binary_data):
        """SSLR(approx=True) 训练后返回准确率列表"""
        from LR import SSLR

        train_X, train_y, test_X, test_y, split_col = _prepare_sslr_data(
            devices, make_binary_data, n_samples=200, n_features=10
        )
        model = SSLR(devices, lambda_=0, approx=True)
        accs = model.fit(
            train_X, train_y,
            X_test=test_X, y_test=test_y,
            n_epochs=3, batch_size=64, val_steps=2, lr=0.1,
        )
        assert isinstance(accs, list)
        assert len(accs) > 0
        # 准确率应在 [0, 1] 范围
        for a in accs:
            assert 0.0 <= float(a) <= 1.0

    def test_fit_non_approx(self, devices, make_binary_data):
        """SSLR(approx=False) 训练，label 在 PYU 上"""
        from LR import SSLR

        company = devices["company"]
        partner = devices["partner"]
        spu = devices["spu"]

        X, y = make_binary_data(n_samples=200, n_features=10, seed=99)
        split_col = 5

        # approx=False 要求 y 在 PYU (非 SPU)
        train_X = sf.to(company, X).to(spu)
        train_y = sf.to(company, y)  # PYU, not SPU

        test_X = load(
            {company: sf.to(company, X[:, :split_col]),
             partner: sf.to(partner, X[:, split_col:])},
            partition_way=PartitionWay.VERTICAL,
        )
        test_y = sf.to(company, y)

        model = SSLR(devices, lambda_=0, approx=False)
        accs = model.fit(
            train_X, train_y,
            X_test=test_X, y_test=test_y,
            n_epochs=2, batch_size=64, val_steps=2, lr=0.1,
        )
        assert len(accs) > 0

    def test_predict_shape(self, devices, make_binary_data):
        """predict 输出形状与样本数一致"""
        from LR import SSLR

        train_X, train_y, test_X, test_y, split_col = _prepare_sslr_data(
            devices, make_binary_data, n_samples=100, n_features=8
        )
        model = SSLR(devices, approx=True)
        model.fit(
            train_X, train_y,
            X_test=test_X, y_test=test_y,
            n_epochs=2, batch_size=32, val_steps=1, lr=0.1,
        )
        y_pred = model.predict(test_X, devices["company"])
        result = sf.reveal(y_pred)
        assert result.shape[0] == 100


# ---------------------------------------------------------------------------
# TC4: SSLR 持久化
# ---------------------------------------------------------------------------

class TestSSLRPersistence:

    def test_save_load_npy(self, devices, make_binary_data):
        """save(npy) → load round-trip"""
        from LR import SSLR

        train_X, train_y, test_X, test_y, split_col = _prepare_sslr_data(
            devices, make_binary_data, n_samples=100, n_features=8
        )
        model = SSLR(devices, approx=True)
        model.fit(
            train_X, train_y,
            X_test=test_X, y_test=test_y,
            n_epochs=2, batch_size=32, val_steps=1, lr=0.1,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = {
                "company": os.path.join(tmpdir, "company_model"),
                "partner": os.path.join(tmpdir, "partner_model"),
            }
            # Pre-create dirs so PYU workers don't race on makedirs
            for p in paths.values():
                os.makedirs(p, exist_ok=True)
            model.save(paths, ext="npy")
            # Barrier: flush PYU queues to ensure async writes complete
            sf.reveal(devices["company"](lambda: 0)())
            sf.reveal(devices["partner"](lambda: 0)())

            # 验证文件存在
            for party in ["company", "partner"]:
                assert os.path.exists(os.path.join(paths[party], "weight.npy"))
                assert os.path.exists(os.path.join(paths[party], "info.json"))

            # load
            loaded = SSLR.load(devices, paths)
            assert loaded.in_features == model.in_features
            assert loaded.out_features == model.out_features

            # 预测结果应一致
            y_orig = sf.reveal(model.predict(test_X, devices["company"]))
            y_loaded = sf.reveal(loaded.predict(test_X, devices["company"]))
            np.testing.assert_array_equal(y_orig, y_loaded)

    def test_save_load_csv(self, devices, make_binary_data):
        """save(csv) → load round-trip"""
        from LR import SSLR

        train_X, train_y, test_X, test_y, split_col = _prepare_sslr_data(
            devices, make_binary_data, n_samples=100, n_features=8
        )
        model = SSLR(devices, approx=True)
        model.fit(
            train_X, train_y,
            X_test=test_X, y_test=test_y,
            n_epochs=2, batch_size=32, val_steps=1, lr=0.1,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = {
                "company": os.path.join(tmpdir, "company_csv"),
                "partner": os.path.join(tmpdir, "partner_csv"),
            }
            for p in paths.values():
                os.makedirs(p, exist_ok=True)
            model.save(paths, ext="csv")
            sf.reveal(devices["company"](lambda: 0)())
            sf.reveal(devices["partner"](lambda: 0)())

            for party in ["company", "partner"]:
                assert os.path.exists(os.path.join(paths[party], "weight.csv"))

            loaded = SSLR.load(devices, paths)
            y_orig = sf.reveal(model.predict(test_X, devices["company"]))
            y_loaded = sf.reveal(loaded.predict(test_X, devices["company"]))
            np.testing.assert_array_equal(y_orig, y_loaded)
