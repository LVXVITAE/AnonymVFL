"""
性能测试 — 分布式模式 (对应 test.tex §5.5 性能测试)

测试内容 (使用 multi_distributed 模式, 两台机器):
- TP1: SSLR 不同样本量 (1K/5K/10K/50K/100K) 训练时间
- TP2: 不同 batch_size 对训练时间的影响
- TP3: SSXGBoost 不同样本量训练时间
- TP4: 推理延迟 (SSLR + SSXGBoost)
- TP7: PSI / SSLR / SSXGBoost 大规模样本压力测试
- WAN: 固定 PSI / 模型参数, 测试不同带宽和延迟条件的影响

所有结果输出到 test_results/ 目录 (CSV + PNG).

注意: 运行前需:
1. 修改 tests/distributed_config.yaml 中的 IP 地址
2. 在 Machine B 上执行 tests/start_partner_worker.sh
3. 执行: pytest -m performance --timeout=0
4. 大规模样本压力测试可单独执行:
   pytest tests/test_performance.py::TestStress -m "performance and slow"
5. WAN 网络影响测试使用独立入口:
   # 使用 tc/netem 模拟带宽和延迟 (需要 sudo 权限).
   PERF_RUN_WAN=1 pytest tests/test_performance.py::TestWANNetworkImpact \
       -m "performance and wan and not slow" --timeout=0
"""
import os
import subprocess
import time
import tracemalloc
import pytest
import yaml
import numpy as np
import secretflow as sf
from secretflow.data.ndarray import load, PartitionWay

_TC_DEVICE = os.getenv("PERF_TC_DEVICE", "lo")
_TC_REMOTE_HOST = os.getenv("PERF_WAN_TC_REMOTE_HOST", "")
_TC_REMOTE_PORT = os.getenv("PERF_WAN_TC_REMOTE_PORT", "22")
_TC_REMOTE_USER = os.getenv("PERF_WAN_TC_REMOTE_USER", "")
_TC_REMOTE_DEVICE = os.getenv("PERF_WAN_TC_REMOTE_DEVICE", "eth0")
_TC_REMOTE_KEY = os.getenv("PERF_WAN_TC_REMOTE_KEY", "")


def _get_net_bytes():
    """获取系统累计网络收发字节数（尽量排除 lo 回环接口）."""
    try:
        import psutil

        net = psutil.net_io_counters(pernic=True)
        tx = 0
        rx = 0
        for nic, counters in net.items():
            if nic == "lo":
                continue
            tx += int(counters.bytes_sent)
            rx += int(counters.bytes_recv)
        # Fallback: if no NIC matched, use aggregate counters.
        if tx == 0 and rx == 0:
            all_net = psutil.net_io_counters(pernic=False)
            tx = int(all_net.bytes_sent)
            rx = int(all_net.bytes_recv)
        return tx, rx
    except Exception:
        # /proc/net/dev fallback to avoid hard dependency on psutil.
        tx = 0
        rx = 0
        try:
            with open("/proc/net/dev", "r", encoding="utf-8") as f:
                lines = f.readlines()[2:]
            for line in lines:
                left, right = line.split(":", 1)
                nic = left.strip()
                if nic == "lo":
                    continue
                fields = right.split()
                if len(fields) >= 9:
                    rx += int(fields[0])
                    tx += int(fields[8])
        except Exception:
            return 0, 0
        return tx, rx


def _run_with_comm_measure(task):
    """执行任务并返回 (result, elapsed_seconds, tx_delta_bytes, rx_delta_bytes)."""
    tx0, rx0 = _get_net_bytes()
    t0 = time.perf_counter()
    result = task()
    elapsed = time.perf_counter() - t0
    tx1, rx1 = _get_net_bytes()
    return result, elapsed, max(0, tx1 - tx0), max(0, rx1 - rx0)

pytestmark = pytest.mark.performance
STRESS_SAMPLES = int(os.getenv("PERF_STRESS_SAMPLES", "300000"))
WAN_CONDITIONS = os.getenv("PERF_WAN_CONDITIONS", "10:0,10:20,10:40,30:0,30:20,30:40,50:0,50:20,50:40")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(TESTS_DIR, "distributed_config.yaml")


def _load_distributed_config():
    if os.getenv("PERF_USE_ENV_CONFIG", "0") == "1":
        return {
            "machine_a": {
                "ip": os.environ["PERF_A_IP"],
                "ray_port": int(os.environ["PERF_A_RAY_PORT"]),
                "object_manager_port": int(os.environ["PERF_A_OBJECT_MANAGER_PORT"]),
                "node_manager_port": int(os.environ["PERF_A_NODE_MANAGER_PORT"]),
                "min_worker_port": int(os.environ["PERF_A_MIN_WORKER_PORT"]),
                "max_worker_port": int(os.environ["PERF_A_MAX_WORKER_PORT"]),
                "company_spu_port": int(os.environ["PERF_A_COMPANY_SPU_PORT"]),
                "coordinator_spu_port": int(os.environ["PERF_A_COORDINATOR_SPU_PORT"]),
                "ray_resources": {"company": 10, "coordinator": 10},
            },
            "machine_b": {
                "ip": os.environ["PERF_B_IP"],
                "partner_spu_port": int(os.environ["PERF_B_PARTNER_SPU_PORT"]),
                "ray_resources": {"partner": 10},
            },
            "ray": {
                "num_cpus": int(os.environ["PERF_RAY_NUM_CPUS"]),
                "object_store_memory": int(os.environ["PERF_RAY_OBJECT_STORE_MEMORY"]),
            },
        }
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)




def _parse_wan_conditions() -> list[tuple[float, int]]:
    conditions = []
    for item in WAN_CONDITIONS.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            bandwidth_text, latency_text = item.split(":", 1)
            conditions.append((float(bandwidth_text), int(float(latency_text))))
        except ValueError as exc:
            raise ValueError(
                "PERF_WAN_CONDITIONS must use comma-separated "
                "'bandwidth_mb_per_s:latency_ms' entries, for example "
                "'10:20,25:50,50:100'."
            ) from exc
    if not conditions:
        raise ValueError("PERF_WAN_CONDITIONS cannot be empty.")
    return conditions


def _wan_condition_id(condition: tuple[float, int]) -> str:
    bandwidth, latency = condition
    bandwidth_text = f"{bandwidth:g}".replace(".", "p")
    return f"{bandwidth_text}Mb_s_{latency}ms"


def _apply_tc(bandwidth_mb_s: float, latency_ms: int, device: str = None):
    if device is None:
        device = _TC_DEVICE
    _clear_tc(device)
    limit = max(100000, latency_ms * 10)
    cmd = (
        f"tc qdisc add dev {device} root handle 1:0 netem delay {latency_ms}ms limit {limit} && "
        f"tc qdisc add dev {device} parent 1:0 handle 2:0 tbf rate {bandwidth_mb_s}mbit burst 32kbit latency 400ms"
    )
    subprocess.run(["sudo", "bash", "-c", cmd], check=True, timeout=10)


def _apply_tc_remote(bandwidth_mb_s: float, latency_ms: int):
    if not _TC_REMOTE_HOST:
        return
    _clear_tc_remote()
    limit = max(100000, latency_ms * 10)
    cmd = (
        f"sudo tc qdisc add dev {_TC_REMOTE_DEVICE} root handle 1:0 netem delay {latency_ms}ms limit {limit} && "
        f"sudo tc qdisc add dev {_TC_REMOTE_DEVICE} parent 1:0 handle 2:0 tbf rate {bandwidth_mb_s}mbit burst 32kbit latency 400ms"
    )
    ssh_args = [
        "ssh", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
        "-p", _TC_REMOTE_PORT,
    ]
    if _TC_REMOTE_KEY:
        ssh_args += ["-i", _TC_REMOTE_KEY]
    ssh_args += [f"{_TC_REMOTE_USER}@{_TC_REMOTE_HOST}", cmd]
    subprocess.run(ssh_args, check=True, timeout=10)


def _clear_tc(device: str = None):
    if device is None:
        device = _TC_DEVICE
    subprocess.run(
        ["sudo", "tc", "qdisc", "del", "dev", device, "root"],
        capture_output=True,
        timeout=10,
    )


def _clear_tc_remote():
    if not _TC_REMOTE_HOST:
        return
    ssh_args = [
        "ssh", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
        "-p", _TC_REMOTE_PORT,
    ]
    if _TC_REMOTE_KEY:
        ssh_args += ["-i", _TC_REMOTE_KEY]
    ssh_args += [f"{_TC_REMOTE_USER}@{_TC_REMOTE_HOST}",
                 f"sudo tc qdisc del dev {_TC_REMOTE_DEVICE} root"]
    subprocess.run(ssh_args, capture_output=True, timeout=10)


def _require_wan_condition(bandwidth_mb_s: float, latency_ms: int):
    if not _TC_REMOTE_HOST and _TC_DEVICE == "lo":
        import warnings
        warnings.warn(
            f"Single-machine mode: skipping tc shaping ({bandwidth_mb_s}Mb/s, {latency_ms}ms) "
            f"on {_TC_DEVICE} to avoid breaking Ray SPU communication."
        )
        return
    _apply_tc(bandwidth_mb_s, latency_ms)
    _apply_tc_remote(bandwidth_mb_s, latency_ms)


def _append_perf_record(perf_results_dir: str, filename: str, record: dict):
    """Append one performance record to a CSV file."""
    import pandas as pd

    csv_path = os.path.join(perf_results_dir, filename)
    df = pd.DataFrame([record])
    df.to_csv(csv_path, mode="a", header=not os.path.exists(csv_path), index=False)


def _build_cluster_def(cfg):
    """从 distributed_config.yaml 构建 SecretFlow cluster_def

    Returns cluster_def
    """
    a = cfg["machine_a"]
    b = cfg["machine_b"]
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
    return cluster_def


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

    cluster_def = _build_cluster_def(cfg)
    ray_addr = f"{cfg['machine_a']['ip']}:{cfg['machine_a']['ray_port']}"

    # Ensure Ray workers can import company/ and root modules.
    company_dir = os.path.join(os.path.dirname(TESTS_DIR), "company")
    project_root = os.path.dirname(TESTS_DIR)
    runtime_env = {"env_vars": {"PYTHONPATH": company_dir + os.pathsep + project_root}}

    mpc = MPCInitializer(
        mode='multi_distributed',
        ray_head_addr=ray_addr,
        cluster_def=cluster_def,
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


def _gen_psi_data_compact(n_company, n_partner, n_features=20,
                          overlap_ratio=0.5, seed=42):
    """生成大规模 PSI 压测数据，避免先构造多份 Python list."""
    rng = np.random.RandomState(seed)
    n_overlap = int(min(n_company, n_partner) * overlap_ratio)

    company_ids = np.arange(n_company, dtype=np.int64)
    partner_ids = np.empty(n_partner, dtype=np.int64)
    partner_ids[:n_overlap] = np.arange(n_overlap, dtype=np.int64)
    partner_ids[n_overlap:] = np.arange(
        n_company,
        n_company + n_partner - n_overlap,
        dtype=np.int64,
    )
    rng.shuffle(company_ids)
    rng.shuffle(partner_ids)

    company_keys = company_ids.astype(str)
    partner_keys = partner_ids.astype(str)

    half = n_features // 2
    company_features = rng.randn(n_company, half).astype(np.float32)
    partner_features = rng.randn(n_partner, half).astype(np.float32)

    return company_keys, company_features, partner_keys, partner_features


# ---------------------------------------------------------------------------
# TP0: PSI 不同样本量对齐时间
# ---------------------------------------------------------------------------

class TestPSIScalability:

    SAMPLE_SIZES = [10000, 20000, 30000, 40000, 50000]

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
            _gen_psi_data(n_samples, n_samples, n_features=18)

        company_data = company(
            lambda k, f: (k, f, None))(company_keys, company_features)
        partner_data = partner(
            lambda k, f: (k, f, None))(partner_keys, partner_features)

        tracemalloc.start()

        def _task():
            R_cI, R_pI, _ = private_set_intersection(
                company_data, partner_data, heu_devices
            )
            # Materialize results to ensure timing is accurate.
            sf.reveal(R_cI)
            sf.reveal(R_pI)

        _, elapsed, tx_bytes, rx_bytes = _run_with_comm_measure(_task)
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        record = {
            "样本数量": n_samples,
            "对齐时间(s)": round(elapsed, 2),
            "内存峰值(MB)": round(peak_mem / 1024 / 1024, 2),
            "发送通信量(MB)": round(tx_bytes / 1024 / 1024, 4),
            "接收通信量(MB)": round(rx_bytes / 1024 / 1024, 4),
            "总通信量(MB)": round((tx_bytes + rx_bytes) / 1024 / 1024, 4),
        }

        _append_perf_record(perf_results_dir, "psi_scalability.csv", record)

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

    SAMPLE_SIZES = [10000, 20000, 30000, 40000, 50000]

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

        _, elapsed, tx_bytes, rx_bytes = _run_with_comm_measure(
            lambda: model.fit(
                train_X,
                train_y,
                X_test=test_X,
                y_test=test_y,
                n_epochs=3,
                batch_size=128,
                val_steps=99999,
                lr=0.1,
            )
        )

        record = {
            "模型": "SSLR",
            "样本数量": n_samples,
            "特征数量": 18,
            "batch_size": 128,
            "n_epochs": 3,
            "总时间(s)": round(elapsed, 2),
            "发送通信量(MB)": round(tx_bytes / 1024 / 1024, 4),
            "接收通信量(MB)": round(rx_bytes / 1024 / 1024, 4),
            "总通信量(MB)": round((tx_bytes + rx_bytes) / 1024 / 1024, 4),
        }

        _append_perf_record(perf_results_dir, "sslr_scalability.csv", record)

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

        n_samples = 10000
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

        _, elapsed, tx_bytes, rx_bytes = _run_with_comm_measure(
            lambda: model.fit(
                train_X,
                train_y,
                X_test=test_X,
                y_test=test_y,
                n_epochs=3,
                batch_size=batch_size,
                val_steps=99999,
                lr=0.1,
            )
        )

        record = {
            "批次大小": batch_size,
            "样本数量": n_samples,
            "n_epochs": 3,
            "总训练时间(s)": round(elapsed, 2),
            "发送通信量(MB)": round(tx_bytes / 1024 / 1024, 4),
            "接收通信量(MB)": round(rx_bytes / 1024 / 1024, 4),
            "总通信量(MB)": round((tx_bytes + rx_bytes) / 1024 / 1024, 4),
        }

        _append_perf_record(perf_results_dir, "batch_size_impact.csv", record)

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

    SAMPLE_SIZES = [10000, 20000, 30000, 40000, 50000]

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

        _, elapsed, tx_bytes, rx_bytes = _run_with_comm_measure(
            lambda: model.fit(train_X, train_y, buckets, FedQuantiles)
        )

        record = {
            "模型": "SSXGBoost",
            "样本数量": n_samples,
            "n_estimators": 3,
            "max_depth": 3,
            "总时间(s)": round(elapsed, 2),
            "发送通信量(MB)": round(tx_bytes / 1024 / 1024, 4),
            "接收通信量(MB)": round(rx_bytes / 1024 / 1024, 4),
            "总通信量(MB)": round((tx_bytes + rx_bytes) / 1024 / 1024, 4),
        }

        _append_perf_record(perf_results_dir, "xgboost_scalability.csv", record)

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

class TestLRInferenceLatency:

    SAMPLE_SIZES = [10000, 20000, 30000, 40000, 50000]
    N_FEATURES = 18
    SPLIT_COL = 9

    @pytest.fixture(scope="class")
    def trained_sslr(self, perf_devices):
        """训练一次 SSLR 模型，供所有推理测试复用"""
        from LR import SSLR

        company = perf_devices["company"]
        spu = perf_devices["spu"]
        X, y = _gen_data(500, n_features=self.N_FEATURES)

        train_X = sf.to(company, X.astype(np.float32)).to(spu)
        train_y = sf.to(company, y.astype(np.float32)).to(spu)

        model = SSLR(perf_devices, approx=True)
        model.fit(train_X, train_y, n_epochs=1, batch_size=128,
                  val_steps=99999, lr=0.1, split_col=self.SPLIT_COL)
        return model

    @pytest.mark.parametrize("n_samples", SAMPLE_SIZES)
    def test_sslr_inference_latency(self, perf_devices, perf_results_dir,
                                    trained_sslr, n_samples):
        company = perf_devices["company"]
        partner = perf_devices["partner"]

        X, _ = _gen_data(n_samples, n_features=self.N_FEATURES)
        test_X = load(
            {company: sf.to(company, X[:, :self.SPLIT_COL].astype(np.float32)),
             partner: sf.to(partner, X[:, self.SPLIT_COL:].astype(np.float32))},
            partition_way=PartitionWay.VERTICAL,
        )

        # 推理计时 + 通信量统计
        _, elapsed, tx_bytes, rx_bytes = _run_with_comm_measure(
            lambda: trained_sslr.predict(test_X, company)
        )

        record = {
            "模型": "SSLR",
            "样本数量": n_samples,
            "推理延迟(s)": round(elapsed, 4),
            "单样本平均延迟(ms)": round(elapsed / n_samples * 1000, 4),
            "发送通信量(MB)": round(tx_bytes / 1024 / 1024, 4),
            "接收通信量(MB)": round(rx_bytes / 1024 / 1024, 4),
            "总通信量(MB)": round((tx_bytes + rx_bytes) / 1024 / 1024, 4),
        }

        _append_perf_record(perf_results_dir, "lr_inference_latency.csv", record)

    def test_lr_inference_latency_plot(self, perf_results_dir):
        from plot_utils import plot_inference_latency
        import pandas as pd

        csv_path = os.path.join(perf_results_dir, "lr_inference_latency.csv")
        if not os.path.exists(csv_path):
            pytest.skip("No inference latency data yet")

        records = pd.read_csv(csv_path).to_dict("records")
        png_path = os.path.join(perf_results_dir, "lr_inference_latency.png")
        plot_inference_latency(records, png_path)
        assert os.path.exists(png_path)


# ---------------------------------------------------------------------------
# TP5: 分位点数量 k 对 SSXGBoost 训练时间的影响
# ---------------------------------------------------------------------------

class TestQuantileImpact:

    K_VALUES = [5, 10, 20, 30, 40, 50]

    @pytest.mark.parametrize("k", K_VALUES)
    def test_quantile_effect(self, perf_devices, perf_results_dir, k):
        """记录不同分位点数量 k 下 SSXGBoost 的训练时间"""
        from XGBoost import SSXGBoost, quantize_buckets, recover_buckets

        company = perf_devices["company"]
        partner = perf_devices["partner"]
        spu = perf_devices["spu"]

        n_samples = 10000
        n_features = 18
        split_col = 9
        X, y = _gen_data(n_samples, n_features=n_features)

        Q1, _, bl1 = quantize_buckets(X[:, :split_col], k=k)
        Q2, _, bl2 = quantize_buckets(X[:, split_col:], k=k)
        buckets = recover_buckets(np.hstack((bl1, bl2)))
        FedQuantiles = load(
            {company: sf.to(company, Q1), partner: sf.to(partner, Q2)},
            partition_way=PartitionWay.HORIZONTAL,
        )

        train_X = sf.to(company, X.astype(np.float32)).to(spu)
        train_y = sf.to(company, y.astype(np.float32))

        model = SSXGBoost(perf_devices, n_estimators=3, max_depth=3)

        _, elapsed, tx_bytes, rx_bytes = _run_with_comm_measure(
            lambda: model.fit(train_X, train_y, buckets, FedQuantiles)
        )

        record = {
            "分位点数量k": k,
            "样本数量": n_samples,
            "n_estimators": 3,
            "max_depth": 3,
            "总训练时间(s)": round(elapsed, 2),
            "发送通信量(MB)": round(tx_bytes / 1024 / 1024, 4),
            "接收通信量(MB)": round(rx_bytes / 1024 / 1024, 4),
            "总通信量(MB)": round((tx_bytes + rx_bytes) / 1024 / 1024, 4),
        }

        _append_perf_record(perf_results_dir, "quantile_impact.csv", record)

    def test_quantile_impact_plot(self, perf_results_dir):
        """读取 CSV 结果并绘图"""
        from plot_utils import plot_time_vs_samples
        import pandas as pd

        csv_path = os.path.join(perf_results_dir, "quantile_impact.csv")
        if not os.path.exists(csv_path):
            pytest.skip("No quantile impact data yet")

        records = pd.read_csv(csv_path).to_dict("records")
        png_path = os.path.join(perf_results_dir, "quantile_impact.png")
        plot_time_vs_samples(records, "分位点数量 k 对 SSXGBoost 训练时间的影响",
                             png_path, x_key="分位点数量k", y_key="总训练时间(s)")
        assert os.path.exists(png_path)


# ---------------------------------------------------------------------------
# TP6: SSXGBoost 推理延迟
# ---------------------------------------------------------------------------

class TestXGBoostInferenceLatency:

    SAMPLE_SIZES = [10000, 20000, 30000, 40000, 50000]
    N_FEATURES = 18
    SPLIT_COL = 9

    @pytest.fixture(scope="class")
    def trained_xgb(self, perf_devices):
        """训练一次 SSXGBoost 模型，供所有推理测试复用"""
        from XGBoost import SSXGBoost, quantize_buckets, recover_buckets

        company = perf_devices["company"]
        partner = perf_devices["partner"]
        spu = perf_devices["spu"]

        X, y = _gen_data(500, n_features=self.N_FEATURES)
        Q1, _, bl1 = quantize_buckets(X[:, :self.SPLIT_COL], k=10)
        Q2, _, bl2 = quantize_buckets(X[:, self.SPLIT_COL:], k=10)
        buckets = recover_buckets(np.hstack((bl1, bl2)))
        FedQuantiles = load(
            {company: sf.to(company, Q1), partner: sf.to(partner, Q2)},
            partition_way=PartitionWay.HORIZONTAL,
        )

        train_X = sf.to(company, X.astype(np.float32)).to(spu)
        train_y = sf.to(company, y.astype(np.float32))

        model = SSXGBoost(perf_devices, n_estimators=3, max_depth=3)
        model.fit(train_X, train_y, buckets, FedQuantiles)
        return model

    @pytest.mark.parametrize("n_samples", SAMPLE_SIZES)
    def test_xgboost_inference_latency(self, perf_devices, perf_results_dir,
                                       trained_xgb, n_samples):
        """记录 SSXGBoost 在不同样本量下的推理延迟"""
        company = perf_devices["company"]
        partner = perf_devices["partner"]

        X, _ = _gen_data(n_samples, n_features=self.N_FEATURES)
        test_X = load(
            {company: sf.to(company, X[:, :self.SPLIT_COL].astype(np.float32)),
             partner: sf.to(partner, X[:, self.SPLIT_COL:].astype(np.float32))},
            partition_way=PartitionWay.VERTICAL,
        )

        _, elapsed, tx_bytes, rx_bytes = _run_with_comm_measure(
            lambda: trained_xgb.predict(test_X, company)
        )

        record = {
            "模型": "SSXGBoost",
            "样本数量": n_samples,
            "推理延迟(s)": round(elapsed, 4),
            "单样本平均延迟(ms)": round(elapsed / n_samples * 1000, 4),
            "发送通信量(MB)": round(tx_bytes / 1024 / 1024, 4),
            "接收通信量(MB)": round(rx_bytes / 1024 / 1024, 4),
            "总通信量(MB)": round((tx_bytes + rx_bytes) / 1024 / 1024, 4),
        }

        _append_perf_record(perf_results_dir, "xgboost_inference_latency.csv", record)

    def test_xgboost_inference_latency_plot(self, perf_results_dir):
        """读取 CSV 结果并绘图"""
        from plot_utils import plot_inference_latency
        import pandas as pd

        csv_path = os.path.join(perf_results_dir, "xgboost_inference_latency.csv")
        if not os.path.exists(csv_path):
            pytest.skip("No XGBoost inference latency data yet")

        records = pd.read_csv(csv_path).to_dict("records")
        png_path = os.path.join(perf_results_dir, "xgboost_inference_latency.png")
        plot_inference_latency(records, png_path)
        assert os.path.exists(png_path)


# ---------------------------------------------------------------------------
# WAN: 固定 PSI / 模型参数, 测试不同带宽和延迟条件的影响
# ---------------------------------------------------------------------------

@pytest.mark.wan
class TestWANNetworkImpact:
    """WAN 网络条件影响测试.

    固定业务负载参数, 只改变网络带宽和延迟:
    - PSI: 10,000 样本, 18 特征, 50% 交集
    - SSLR: 10,000 样本, 18 特征, batch_size=128, n_epochs=3
    - SSXGBoost: 10,000 样本, 18 特征, k=20, n_estimators=3, max_depth=3

    PERF_WAN_CONDITIONS 使用 "带宽Mb/s:延迟ms" 列表, 例如:
    PERF_WAN_CONDITIONS=10:20,25:50,50:100
    """

    N_SAMPLES = 10000
    N_FEATURES = 18
    SPLIT_COL = 9
    PSI_RESULT_FILE = "psi_network_impact_wan.csv"
    SSLR_RESULT_FILE = "sslr_network_impact_wan.csv"
    XGB_RESULT_FILE = "xgboost_network_impact_wan.csv"

    @pytest.fixture(autouse=True)
    def _check_wan_enabled(self):
        if os.getenv("PERF_RUN_WAN", "0") != "1":
            pytest.skip("Set PERF_RUN_WAN=1 to run WAN performance tests.")

    @pytest.fixture(autouse=True)
    def _tc_cleanup(self):
        yield
        _clear_tc()
        _clear_tc_remote()

    @staticmethod
    def _condition_record(condition: tuple[float, int]) -> dict:
        bandwidth, latency = condition
        return {
            "网络环境": "WAN",
            "带宽限制(Mb/s)": bandwidth,
            "延迟(ms)": latency,
        }

    @staticmethod
    def _append_record(perf_results_dir: str, filename: str, record: dict):
        _append_perf_record(perf_results_dir, filename, record)

    @pytest.mark.parametrize(
        "wan_condition",
        _parse_wan_conditions(),
        ids=_wan_condition_id,
    )
    def test_psi_network_impact(self, perf_devices, perf_results_dir, wan_condition):
        """固定 PSI 数据规模, 记录不同 WAN 条件下的对齐时间."""
        from PSI import private_set_intersection

        bandwidth, latency = wan_condition
        _require_wan_condition(bandwidth, latency)

        company = perf_devices["company"]
        partner = perf_devices["partner"]
        heu_devices = (perf_devices["company_heu"], perf_devices["partner_heu"])

        company_keys, company_features, partner_keys, partner_features = _gen_psi_data(
            self.N_SAMPLES,
            self.N_SAMPLES,
            n_features=self.N_FEATURES,
            overlap_ratio=0.5,
            seed=3030,
        )
        company_data = company(lambda k, f: (k, f, None))(company_keys, company_features)
        partner_data = partner(lambda k, f: (k, f, None))(partner_keys, partner_features)

        tracemalloc.start()

        def _task():
            R_cI, R_pI, _ = private_set_intersection(
                company_data, partner_data, heu_devices
            )
            sf.reveal(R_cI)
            sf.reveal(R_pI)

        _, elapsed, tx_bytes, rx_bytes = _run_with_comm_measure(_task)
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        record = {
            **self._condition_record(wan_condition),
            "测试": "PSI-WAN-NetworkImpact",
            "样本数量": self.N_SAMPLES,
            "特征数量": self.N_FEATURES,
            "交集比例": 0.5,
            "对齐时间(s)": round(elapsed, 2),
            "内存峰值(MB)": round(peak_mem / 1024 / 1024, 2),
            "发送通信量(MB)": round(tx_bytes / 1024 / 1024, 4),
            "接收通信量(MB)": round(rx_bytes / 1024 / 1024, 4),
            "总通信量(MB)": round((tx_bytes + rx_bytes) / 1024 / 1024, 4),
        }
        self._append_record(perf_results_dir, self.PSI_RESULT_FILE, record)

    @pytest.mark.parametrize(
        "wan_condition",
        _parse_wan_conditions(),
        ids=_wan_condition_id,
    )
    def test_sslr_network_impact(self, perf_devices, perf_results_dir, wan_condition):
        """固定 SSLR 模型参数, 记录不同 WAN 条件下的训练时间."""
        from LR import SSLR

        bandwidth, latency = wan_condition
        _require_wan_condition(bandwidth, latency)

        company = perf_devices["company"]
        partner = perf_devices["partner"]
        spu = perf_devices["spu"]

        X, y = _gen_data(self.N_SAMPLES, n_features=self.N_FEATURES, seed=3031)
        train_X = sf.to(company, X).to(spu)
        train_y = sf.to(company, y).to(spu)
        test_X = load(
            {company: sf.to(company, X[:200, :self.SPLIT_COL]),
             partner: sf.to(partner, X[:200, self.SPLIT_COL:])},
            partition_way=PartitionWay.VERTICAL,
        )
        test_y = sf.to(company, y[:200])

        batch_size = 128
        n_epochs = 3
        model = SSLR(perf_devices, approx=True)
        _, elapsed, tx_bytes, rx_bytes = _run_with_comm_measure(
            lambda: model.fit(
                train_X,
                train_y,
                X_test=test_X,
                y_test=test_y,
                n_epochs=n_epochs,
                batch_size=batch_size,
                val_steps=99999,
                lr=0.1,
            )
        )

        record = {
            **self._condition_record(wan_condition),
            "测试": "SSLR-WAN-NetworkImpact",
            "模型": "SSLR",
            "样本数量": self.N_SAMPLES,
            "特征数量": self.N_FEATURES,
            "batch_size": batch_size,
            "n_epochs": n_epochs,
            "总时间(s)": round(elapsed, 2),
            "发送通信量(MB)": round(tx_bytes / 1024 / 1024, 4),
            "接收通信量(MB)": round(rx_bytes / 1024 / 1024, 4),
            "总通信量(MB)": round((tx_bytes + rx_bytes) / 1024 / 1024, 4),
        }
        self._append_record(perf_results_dir, self.SSLR_RESULT_FILE, record)

    @pytest.mark.parametrize(
        "wan_condition",
        _parse_wan_conditions(),
        ids=_wan_condition_id,
    )
    def test_xgboost_network_impact(self, perf_devices, perf_results_dir, wan_condition):
        """固定 SSXGBoost 模型参数, 记录不同 WAN 条件下的训练时间."""
        from XGBoost import SSXGBoost, quantize_buckets, recover_buckets

        bandwidth, latency = wan_condition
        _require_wan_condition(bandwidth, latency)

        company = perf_devices["company"]
        partner = perf_devices["partner"]
        spu = perf_devices["spu"]

        X, y = _gen_data(self.N_SAMPLES, n_features=self.N_FEATURES, seed=3032)
        k_quantiles = 20
        n_estimators = 3
        max_depth = 3

        Q1, _, bl1 = quantize_buckets(X[:, :self.SPLIT_COL], k=k_quantiles)
        Q2, _, bl2 = quantize_buckets(X[:, self.SPLIT_COL:], k=k_quantiles)
        buckets = recover_buckets(np.hstack((bl1, bl2)))
        FedQuantiles = load(
            {company: sf.to(company, Q1), partner: sf.to(partner, Q2)},
            partition_way=PartitionWay.HORIZONTAL,
        )
        train_X = sf.to(company, X.astype(np.float32)).to(spu)
        train_y = sf.to(company, y.astype(np.float32))

        model = SSXGBoost(
            perf_devices,
            n_estimators=n_estimators,
            max_depth=max_depth,
        )
        _, elapsed, tx_bytes, rx_bytes = _run_with_comm_measure(
            lambda: model.fit(train_X, train_y, buckets, FedQuantiles)
        )

        record = {
            **self._condition_record(wan_condition),
            "测试": "SSXGBoost-WAN-NetworkImpact",
            "模型": "SSXGBoost",
            "样本数量": self.N_SAMPLES,
            "特征数量": self.N_FEATURES,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "分位点数量k": k_quantiles,
            "总时间(s)": round(elapsed, 2),
            "发送通信量(MB)": round(tx_bytes / 1024 / 1024, 4),
            "接收通信量(MB)": round(rx_bytes / 1024 / 1024, 4),
            "总通信量(MB)": round((tx_bytes + rx_bytes) / 1024 / 1024, 4),
        }
        self._append_record(perf_results_dir, self.XGB_RESULT_FILE, record)

    def test_psi_network_impact_plot(self, perf_results_dir):
        from plot_utils import plot_wan_network_impact
        import pandas as pd

        csv_path = os.path.join(perf_results_dir, self.PSI_RESULT_FILE)
        if not os.path.exists(csv_path):
            pytest.skip("No PSI WAN network impact data yet")
        records = pd.read_csv(csv_path).to_dict("records")
        png_path = os.path.join(perf_results_dir, "psi_network_impact_wan.png")
        plot_wan_network_impact(
            records, "WAN 条件对 PSI 对齐时间的影响", png_path, "对齐时间(s)"
        )
        assert os.path.exists(png_path)

    def test_sslr_network_impact_plot(self, perf_results_dir):
        from plot_utils import plot_wan_network_impact
        import pandas as pd

        csv_path = os.path.join(perf_results_dir, self.SSLR_RESULT_FILE)
        if not os.path.exists(csv_path):
            pytest.skip("No SSLR WAN network impact data yet")
        records = pd.read_csv(csv_path).to_dict("records")
        png_path = os.path.join(perf_results_dir, "sslr_network_impact_wan.png")
        plot_wan_network_impact(
            records, "WAN 条件对 SSLR 训练时间的影响", png_path, "总时间(s)"
        )
        assert os.path.exists(png_path)

    def test_xgboost_network_impact_plot(self, perf_results_dir):
        from plot_utils import plot_wan_network_impact
        import pandas as pd

        csv_path = os.path.join(perf_results_dir, self.XGB_RESULT_FILE)
        if not os.path.exists(csv_path):
            pytest.skip("No SSXGBoost WAN network impact data yet")
        records = pd.read_csv(csv_path).to_dict("records")
        png_path = os.path.join(perf_results_dir, "xgboost_network_impact_wan.png")
        plot_wan_network_impact(
            records, "WAN 条件对 SSXGBoost 训练时间的影响", png_path, "总时间(s)"
        )
        assert os.path.exists(png_path)


# ---------------------------------------------------------------------------
# TP7: 大规模样本压力测试
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestStress:
    """PSI、SSLR、SSXGBoost 的大规模样本压力测试.

    默认样本量为 300,000；如需在本地先做冒烟验证，可临时设置:
    PERF_STRESS_SAMPLES=10000 pytest tests/test_performance.py::TestStress ...
    """

    N_SAMPLES = STRESS_SAMPLES
    N_FEATURES = 18
    SPLIT_COL = 9

    @staticmethod
    def _append_record(perf_results_dir, filename, record):
        import pandas as pd

        csv_path = os.path.join(perf_results_dir, filename)
        df = pd.DataFrame([record])
        df.to_csv(csv_path, mode="a", header=not os.path.exists(csv_path), index=False)

    def test_psi_alignment_stress(self, perf_devices, perf_results_dir):
        """PSI 大规模样本对齐压力测试."""
        from PSI import private_set_intersection

        company = perf_devices["company"]
        partner = perf_devices["partner"]
        heu_devices = (perf_devices["company_heu"], perf_devices["partner_heu"])

        company_keys, company_features, partner_keys, partner_features = _gen_psi_data_compact(
            self.N_SAMPLES,
            self.N_SAMPLES,
            n_features=self.N_FEATURES,
            overlap_ratio=0.5,
            seed=2026,
        )

        company_data = company(
            lambda k, f: (k, f, None))(company_keys, company_features)
        partner_data = partner(
            lambda k, f: (k, f, None))(partner_keys, partner_features)

        tracemalloc.start()

        def _task():
            R_cI, R_pI, _ = private_set_intersection(
                company_data, partner_data, heu_devices
            )
            sf.reveal(R_cI)
            sf.reveal(R_pI)

        _, elapsed, tx_bytes, rx_bytes = _run_with_comm_measure(_task)
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        record = {
            "测试": "PSI-Stress",
            "样本数量": self.N_SAMPLES,
            "特征数量": self.N_FEATURES,
            "交集比例": 0.5,
            "对齐时间(s)": round(elapsed, 2),
            "内存峰值(MB)": round(peak_mem / 1024 / 1024, 2),
            "发送通信量(MB)": round(tx_bytes / 1024 / 1024, 4),
            "接收通信量(MB)": round(rx_bytes / 1024 / 1024, 4),
            "总通信量(MB)": round((tx_bytes + rx_bytes) / 1024 / 1024, 4),
        }
        self._append_record(perf_results_dir, "psi_stress.csv", record)

    def test_sslr_training_stress(self, perf_devices, perf_results_dir):
        """SSLR 大规模样本训练压力测试."""
        from LR import SSLR

        company = perf_devices["company"]
        spu = perf_devices["spu"]

        X, y = _gen_data(self.N_SAMPLES, n_features=self.N_FEATURES, seed=2027)
        train_X = sf.to(company, X.astype(np.float32)).to(spu)
        train_y = sf.to(company, y.astype(np.float32)).to(spu)

        model = SSLR(perf_devices, approx=True)
        batch_size = 8192
        n_epochs = 1

        tracemalloc.start()
        _, elapsed, tx_bytes, rx_bytes = _run_with_comm_measure(
            lambda: model.fit(
                train_X,
                train_y,
                n_epochs=n_epochs,
                batch_size=batch_size,
                val_steps=999999999,
                lr=0.1,
                split_col=self.SPLIT_COL,
            )
        )
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        record = {
            "测试": "SSLR-Stress",
            "模型": "SSLR",
            "样本数量": self.N_SAMPLES,
            "特征数量": self.N_FEATURES,
            "batch_size": batch_size,
            "n_epochs": n_epochs,
            "总时间(s)": round(elapsed, 2),
            "内存峰值(MB)": round(peak_mem / 1024 / 1024, 2),
            "发送通信量(MB)": round(tx_bytes / 1024 / 1024, 4),
            "接收通信量(MB)": round(rx_bytes / 1024 / 1024, 4),
            "总通信量(MB)": round((tx_bytes + rx_bytes) / 1024 / 1024, 4),
        }
        self._append_record(perf_results_dir, "sslr_stress.csv", record)

    def test_xgboost_training_stress(self, perf_devices, perf_results_dir):
        """SSXGBoost 大规模样本训练压力测试."""
        from XGBoost import SSXGBoost, quantize_buckets, recover_buckets

        company = perf_devices["company"]
        partner = perf_devices["partner"]
        spu = perf_devices["spu"]

        X, y = _gen_data(self.N_SAMPLES, n_features=self.N_FEATURES, seed=2028)
        k_quantiles = 10
        n_estimators = 1
        max_depth = 2

        Q1, _, bl1 = quantize_buckets(X[:, :self.SPLIT_COL], k=k_quantiles)
        Q2, _, bl2 = quantize_buckets(X[:, self.SPLIT_COL:], k=k_quantiles)
        buckets = recover_buckets(np.hstack((bl1, bl2)))
        FedQuantiles = load(
            {company: sf.to(company, Q1), partner: sf.to(partner, Q2)},
            partition_way=PartitionWay.HORIZONTAL,
        )

        train_X = sf.to(company, X.astype(np.float32)).to(spu)
        train_y = sf.to(company, y.astype(np.float32))
        model = SSXGBoost(
            perf_devices,
            n_estimators=n_estimators,
            max_depth=max_depth,
        )

        tracemalloc.start()
        _, elapsed, tx_bytes, rx_bytes = _run_with_comm_measure(
            lambda: model.fit(train_X, train_y, buckets, FedQuantiles)
        )
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        record = {
            "测试": "SSXGBoost-Stress",
            "模型": "SSXGBoost",
            "样本数量": self.N_SAMPLES,
            "特征数量": self.N_FEATURES,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "分位点数量k": k_quantiles,
            "总时间(s)": round(elapsed, 2),
            "内存峰值(MB)": round(peak_mem / 1024 / 1024, 2),
            "发送通信量(MB)": round(tx_bytes / 1024 / 1024, 4),
            "接收通信量(MB)": round(rx_bytes / 1024 / 1024, 4),
            "总通信量(MB)": round((tx_bytes + rx_bytes) / 1024 / 1024, 4),
        }
        self._append_record(
            perf_results_dir, "xgboost_stress.csv", record
        )
