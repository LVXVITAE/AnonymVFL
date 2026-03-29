"""
性能测试 — 分布式模式 (对应 test.tex §5.5 性能测试)

测试内容 (使用 multi_distributed 模式, 两台机器):
- TP1: SSLR 不同样本量 (1K/5K/10K/50K/100K) 训练时间
- TP2: 不同 batch_size 对训练时间的影响
- TP3: SSXGBoost 不同样本量训练时间
- TP4: 推理延迟 (SSLR + SSXGBoost)

所有结果输出到 test_results/ 目录 (CSV + PNG).

注意: 运行前需:
1. 修改 tests/distributed_config.yaml 中的 IP 地址
2. 在 Machine B 上执行 tests/start_partner_worker.sh
3. 执行: pytest -m performance --timeout=0
"""
import os
import time
import pytest
import yaml
import numpy as np
import secretflow as sf
from secretflow.data.ndarray import load, PartitionWay

pytestmark = pytest.mark.performance

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(TESTS_DIR, "distributed_config.yaml")


def _load_distributed_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def _build_cluster_def(cfg):
    """从 distributed_config.yaml 构建 SecretFlow cluster_def."""
    a = cfg["machine_a"]
    b = cfg["machine_b"]
    return {
        "nodes": [
            {
                "party": "company",
                "address": f"{a['ip']}:{a['company_spu_port']}",
                "listen_addr": f"0.0.0.0:{a['company_spu_port']}",
            },
            {
                "party": "partner",
                "address": f"{b['ip']}:{b['partner_spu_port']}",
                "listen_addr": f"0.0.0.0:{b['partner_spu_port']}",
            },
            {
                "party": "coordinator",
                "address": f"{a['ip']}:{a['coordinator_spu_port']}",
                "listen_addr": f"0.0.0.0:{a['coordinator_spu_port']}",
            },
        ],
        "runtime_config": {"protocol": 3, "field": 3},
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def distributed_env():
    """初始化 multi_distributed 环境.

    如果配置文件中 IP 仍为占位符, 退化为 single_sim 模式并发出警告.
    """
    cfg = _load_distributed_config()
    is_placeholder = "MACHINE" in cfg["machine_a"]["ip"] or "MACHINE" in cfg["machine_b"]["ip"]

    if is_placeholder:
        pytest.skip(
            "distributed_config.yaml 中 IP 为占位符, 请配置实际 IP 后运行性能测试. "
            "如需单机退化测试, 将两个 IP 均设为 127.0.0.1"
        )

    try:
        sf.shutdown()
    except Exception:
        pass

    from common import MPCInitializer
    cluster_def = _build_cluster_def(cfg)
    ray_addr = f"{cfg['machine_a']['ip']}:{cfg['machine_a']['ray_port']}"
    mpc = MPCInitializer(
        mode="multi_distributed",
        ray_head_addr=ray_addr,
        cluster_def=cluster_def,
    )
    yield mpc, cfg
    try:
        sf.shutdown()
    except Exception:
        pass


@pytest.fixture(scope="module")
def perf_devices(distributed_env):
    mpc, _ = distributed_env
    return {
        "spu": mpc.spu,
        "company": mpc.company,
        "partner": mpc.partner,
        "coordinator": mpc.coordinator,
    }


@pytest.fixture(scope="module")
def perf_results_dir(test_results_dir):
    d = os.path.join(test_results_dir, "performance")
    os.makedirs(d, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------

def _gen_data(n_samples, n_features=18, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features).astype(np.float32)
    w = rng.randn(n_features, 1).astype(np.float32)
    y = (1.0 / (1.0 + np.exp(-(X @ w))) > 0.5).astype(np.float32)
    return X, y


# ---------------------------------------------------------------------------
# TP1: SSLR 不同样本量训练时间
# ---------------------------------------------------------------------------

class TestSSLRScalability:

    SAMPLE_SIZES = [1000, 5000, 10000, 50000, 100000]

    @pytest.mark.parametrize("n_samples", SAMPLE_SIZES)
    def test_sslr_training_time(self, perf_devices, perf_results_dir, n_samples):
        """记录 SSLR 在不同样本量下的训练时间"""
        from LR import SSLR

        company = perf_devices["company"]
        partner = perf_devices["partner"]
        spu = perf_devices["spu"]

        X, y = _gen_data(n_samples, n_features=18)
        split_col = 9

        train_X = sf.to(company, X).to(spu)
        train_y = sf.to(company, y).to(spu)

        test_X = load(
            {company: sf.to(company, X[:200, :split_col]),
             partner: sf.to(partner, X[:200, split_col:])},
            partition_way=PartitionWay.VERTICAL,
        )
        test_y = sf.to(company, y[:200])

        model = SSLR(perf_devices, approx=True)

        t0 = time.time()
        model.fit(train_X, train_y, X_test=test_X, y_test=test_y,
                  n_epochs=3, batch_size=128, val_steps=99999, lr=0.1)
        elapsed = time.time() - t0

        record = {
            "模型": "SSLR",
            "样本数量": n_samples,
            "特征数量": 18,
            "batch_size": 128,
            "n_epochs": 3,
            "总时间(s)": round(elapsed, 2),
        }

        # 追加到 CSV
        csv_path = os.path.join(perf_results_dir, "sslr_scalability.csv")
        import pandas as pd
        df = pd.DataFrame([record])
        df.to_csv(csv_path, mode="a", header=not os.path.exists(csv_path), index=False)

    def test_sslr_scalability_plot(self, perf_results_dir):
        """读取 CSV 结果并绘图"""
        from plot_utils import plot_time_vs_samples
        import pandas as pd

        csv_path = os.path.join(perf_results_dir, "sslr_scalability.csv")
        if not os.path.exists(csv_path):
            pytest.skip("No SSLR scalability data yet")

        records = pd.read_csv(csv_path).to_dict("records")
        png_path = os.path.join(perf_results_dir, "sslr_scalability.png")
        plot_time_vs_samples(records, "SSLR 训练时间 vs 样本数量", png_path)
        assert os.path.exists(png_path)


# ---------------------------------------------------------------------------
# TP2: Batch size 对 SSLR 训练时间的影响
# ---------------------------------------------------------------------------

class TestBatchSizeImpact:

    BATCH_SIZES = [32, 64, 128, 256, 512, 1024]

    @pytest.mark.parametrize("batch_size", BATCH_SIZES)
    def test_batch_size_effect(self, perf_devices, perf_results_dir, batch_size):
        from LR import SSLR

        company = perf_devices["company"]
        partner = perf_devices["partner"]
        spu = perf_devices["spu"]

        n_samples = 5000
        X, y = _gen_data(n_samples)
        split_col = 9

        train_X = sf.to(company, X).to(spu)
        train_y = sf.to(company, y).to(spu)

        test_X = load(
            {company: sf.to(company, X[:200, :split_col]),
             partner: sf.to(partner, X[:200, split_col:])},
            partition_way=PartitionWay.VERTICAL,
        )
        test_y = sf.to(company, y[:200])

        model = SSLR(perf_devices, approx=True)

        t0 = time.time()
        model.fit(train_X, train_y, X_test=test_X, y_test=test_y,
                  n_epochs=3, batch_size=batch_size, val_steps=99999, lr=0.1)
        elapsed = time.time() - t0

        record = {
            "批次大小": batch_size,
            "样本数量": n_samples,
            "n_epochs": 3,
            "总训练时间(s)": round(elapsed, 2),
        }

        csv_path = os.path.join(perf_results_dir, "batch_size_impact.csv")
        import pandas as pd
        df = pd.DataFrame([record])
        df.to_csv(csv_path, mode="a", header=not os.path.exists(csv_path), index=False)

    def test_batch_size_plot(self, perf_results_dir):
        from plot_utils import plot_batch_size_impact
        import pandas as pd

        csv_path = os.path.join(perf_results_dir, "batch_size_impact.csv")
        if not os.path.exists(csv_path):
            pytest.skip("No batch size data yet")

        records = pd.read_csv(csv_path).to_dict("records")
        png_path = os.path.join(perf_results_dir, "batch_size_impact.png")
        plot_batch_size_impact(records, png_path)
        assert os.path.exists(png_path)


# ---------------------------------------------------------------------------
# TP3: SSXGBoost 不同样本量训练时间
# ---------------------------------------------------------------------------

class TestSSXGBoostScalability:

    SAMPLE_SIZES = [1000, 5000, 10000]

    @pytest.mark.parametrize("n_samples", SAMPLE_SIZES)
    def test_xgboost_training_time(self, perf_devices, perf_results_dir, n_samples):
        from XGBoost import SSXGBoost, quantize_buckets, recover_buckets

        company = perf_devices["company"]
        partner = perf_devices["partner"]
        spu = perf_devices["spu"]

        X, y = _gen_data(n_samples, n_features=18)
        split_col = 9

        Q1, _, bl1 = quantize_buckets(X[:, :split_col], k=20)
        Q2, _, bl2 = quantize_buckets(X[:, split_col:], k=20)
        buckets = recover_buckets(np.hstack((bl1, bl2)))
        FedQuantiles = load(
            {company: sf.to(company, Q1), partner: sf.to(partner, Q2)},
            partition_way=PartitionWay.HORIZONTAL,
        )

        train_X = sf.to(company, X.astype(np.float32)).to(spu)
        train_y = sf.to(company, y.astype(np.float32))

        model = SSXGBoost(perf_devices, n_estimators=3, max_depth=3)

        t0 = time.time()
        model.fit(train_X, train_y, buckets, FedQuantiles)
        elapsed = time.time() - t0

        record = {
            "模型": "SSXGBoost",
            "样本数量": n_samples,
            "n_estimators": 3,
            "max_depth": 3,
            "总时间(s)": round(elapsed, 2),
        }

        csv_path = os.path.join(perf_results_dir, "xgboost_scalability.csv")
        import pandas as pd
        df = pd.DataFrame([record])
        df.to_csv(csv_path, mode="a", header=not os.path.exists(csv_path), index=False)

    def test_xgboost_scalability_plot(self, perf_results_dir):
        from plot_utils import plot_time_vs_samples
        import pandas as pd

        csv_path = os.path.join(perf_results_dir, "xgboost_scalability.csv")
        if not os.path.exists(csv_path):
            pytest.skip("No XGBoost scalability data yet")

        records = pd.read_csv(csv_path).to_dict("records")
        png_path = os.path.join(perf_results_dir, "xgboost_scalability.png")
        plot_time_vs_samples(records, "SSXGBoost 训练时间 vs 样本数量", png_path)
        assert os.path.exists(png_path)


# ---------------------------------------------------------------------------
# TP4: 推理延迟
# ---------------------------------------------------------------------------

class TestInferenceLatency:

    SAMPLE_SIZES = [100, 500, 1000, 5000, 10000]

    @pytest.mark.parametrize("n_samples", SAMPLE_SIZES)
    def test_sslr_inference_latency(self, perf_devices, perf_results_dir, n_samples):
        from LR import SSLR

        company = perf_devices["company"]
        partner = perf_devices["partner"]
        spu = perf_devices["spu"]

        n_features = 18
        split_col = 9
        X, y = _gen_data(n_samples, n_features=n_features)

        # 快速训练
        train_X = sf.to(company, X[:500].astype(np.float32)).to(spu)
        train_y = sf.to(company, y[:500].astype(np.float32)).to(spu)
        test_X = load(
            {company: sf.to(company, X[:, :split_col].astype(np.float32)),
             partner: sf.to(partner, X[:, split_col:].astype(np.float32))},
            partition_way=PartitionWay.VERTICAL,
        )

        model = SSLR(perf_devices, approx=True)
        model.fit(train_X, train_y, n_epochs=1, batch_size=128,
                  val_steps=99999, lr=0.1, split_col=split_col)

        # 推理计时
        t0 = time.time()
        model.predict(test_X, company)
        elapsed = time.time() - t0

        record = {
            "模型": "SSLR",
            "样本数量": n_samples,
            "推理延迟(s)": round(elapsed, 4),
            "单样本平均延迟(ms)": round(elapsed / n_samples * 1000, 4),
        }

        csv_path = os.path.join(perf_results_dir, "inference_latency.csv")
        import pandas as pd
        df = pd.DataFrame([record])
        df.to_csv(csv_path, mode="a", header=not os.path.exists(csv_path), index=False)

    def test_inference_latency_plot(self, perf_results_dir):
        from plot_utils import plot_inference_latency
        import pandas as pd

        csv_path = os.path.join(perf_results_dir, "inference_latency.csv")
        if not os.path.exists(csv_path):
            pytest.skip("No inference latency data yet")

        records = pd.read_csv(csv_path).to_dict("records")
        png_path = os.path.join(perf_results_dir, "inference_latency.png")
        plot_inference_latency(records, png_path)
        assert os.path.exists(png_path)
