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
import tracemalloc
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
    """从 distributed_config.yaml 构建 SecretFlow cluster_def 和 link_desc.

    Returns (cluster_def, link_desc) — link_desc must be passed as a
    **separate** keyword argument to ``sf.SPU()``.
    """
    a = cfg["machine_a"]
    b = cfg["machine_b"]
    link_cfg = cfg.get("link_desc", {})
    cluster_def = {
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
    link_desc = {
        "connect_retry_times": link_cfg.get("connect_retry_times", 60),
        "connect_retry_interval_ms": link_cfg.get("connect_retry_interval_ms", 2000),
        "recv_timeout_ms": link_cfg.get("recv_timeout_ms", 300000),
    }
    return cluster_def, link_desc


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def distributed_env():
    """初始化 multi_distributed 环境 (通过 MPCInitializer).

    使用 MPCInitializer 初始化 SPU / PYU / HEU 设备,
    以支持 PSI 等需要 HEU 的性能测试.

    如果配置文件中 IP 仍为占位符, 跳过测试.
    """
    from common import MPCInitializer

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

    cluster_def, link_desc = _build_cluster_def(cfg)
    ray_addr = f"{cfg['machine_a']['ip']}:{cfg['machine_a']['ray_port']}"

    # Ensure Ray workers can import company/ modules.
    company_dir = os.path.join(os.path.dirname(TESTS_DIR), "company")
    runtime_env = {"env_vars": {"PYTHONPATH": company_dir}}

    mpc = MPCInitializer(
        mode='multi_distributed',
        ray_head_addr=ray_addr,
        cluster_def=cluster_def,
        link_desc=link_desc,
        runtime_env=runtime_env,
    )

    env = type("Env", (), {
        "spu": mpc.spu, "company": mpc.company,
        "partner": mpc.partner, "coordinator": mpc.coordinator,
        "company_heu": mpc.company_heu, "partner_heu": mpc.partner_heu,
    })()
    yield env, cfg
    try:
        sf.shutdown()
    except Exception:
        pass


@pytest.fixture(scope="module")
def perf_devices(distributed_env):
    env, _ = distributed_env
    return {
        "spu": env.spu,
        "company": env.company,
        "partner": env.partner,
        "coordinator": env.coordinator,
        "company_heu": env.company_heu,
        "partner_heu": env.partner_heu,
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


def _gen_psi_data(n_company, n_partner, n_features=20, overlap_ratio=0.5, seed=42):
    """生成 PSI 性能测试用的合成数据.

    每方各持有 n_features//2 维私有特征, 无公开特征.
    overlap_ratio 控制两方 key 的交集比例.
    """
    rng = np.random.RandomState(seed)
    n_overlap = int(min(n_company, n_partner) * overlap_ratio)
    shared_ids = list(range(n_overlap))
    company_only = list(range(n_overlap, n_overlap + n_company - n_overlap))
    partner_only = list(range(n_overlap + n_company - n_overlap,
                              n_overlap + n_company - n_overlap + n_partner - n_overlap))

    company_keys = [str(k) for k in (shared_ids + company_only)]
    partner_keys = [str(k) for k in (shared_ids + partner_only)]
    rng.shuffle(company_keys)
    rng.shuffle(partner_keys)

    half = n_features // 2
    company_features = rng.randn(n_company, half).astype(np.float32)
    partner_features = rng.randn(n_partner, half).astype(np.float32)

    return company_keys, company_features, partner_keys, partner_features


# ---------------------------------------------------------------------------
# TP0: PSI 不同样本量对齐时间
# ---------------------------------------------------------------------------

class TestPSIScalability:

    SAMPLE_SIZES = [1000, 5000, 10000, 50000, 100000]

    @pytest.mark.parametrize("n_samples", SAMPLE_SIZES)
    def test_psi_alignment_time(self, perf_devices, perf_results_dir, n_samples):
        """记录 PSI 在不同样本量下的对齐时间、内存占用"""
        from PSI import private_set_intersection

        company = perf_devices["company"]
        partner = perf_devices["partner"]
        company_heu = perf_devices["company_heu"]
        partner_heu = perf_devices["partner_heu"]
        heu_devices = (company_heu, partner_heu)

        company_keys, company_features, partner_keys, partner_features = \
            _gen_psi_data(n_samples, n_samples, n_features=20)

        company_data = company(
            lambda k, f: (k, f, None))(company_keys, company_features)
        partner_data = partner(
            lambda k, f: (k, f, None))(partner_keys, partner_features)

        tracemalloc.start()
        t0 = time.time()
        R_cI, R_pI, bucket_labels = private_set_intersection(
            company_data, partner_data, heu_devices)
        # Materialize results to ensure timing is accurate.
        sf.reveal(R_cI)
        sf.reveal(R_pI)
        elapsed = time.time() - t0
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        record = {
            "样本数量": n_samples,
            "对齐时间(s)": round(elapsed, 2),
            "内存峰值(MB)": round(peak_mem / 1024 / 1024, 2),
        }

        csv_path = os.path.join(perf_results_dir, "psi_scalability.csv")
        import pandas as pd
        df = pd.DataFrame([record])
        df.to_csv(csv_path, mode="a", header=not os.path.exists(csv_path), index=False)

    def test_psi_scalability_plot(self, perf_results_dir):
        """读取 CSV 结果并绘图"""
        from plot_utils import plot_time_vs_samples
        import pandas as pd

        csv_path = os.path.join(perf_results_dir, "psi_scalability.csv")
        if not os.path.exists(csv_path):
            pytest.skip("No PSI scalability data yet")

        records = pd.read_csv(csv_path).to_dict("records")
        png_path = os.path.join(perf_results_dir, "psi_scalability.png")
        plot_time_vs_samples(records, "PSI 对齐时间 vs 样本数量", png_path,
                             y_key="对齐时间(s)")
        assert os.path.exists(png_path)


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
