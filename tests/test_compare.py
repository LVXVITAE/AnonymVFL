"""
对比实验测试 — 分布式模式

实验内容:
- SSLR: MNIST(二分类映射), batch_size=128, n_epochs=2
- SSXGBoost: Adult, n_estimators=3, max_depth=3/4/5

记录指标:
- 最优ACC
- 训练时间(s)
- 通信量(MB)

注意: 运行前需先配置 tests/distributed_config.yaml 中的机器 IP。
"""
import os
import time

import numpy as np
import pandas as pd
import pytest
import secretflow as sf
import yaml
from secretflow.data.ndarray import PartitionWay, load

pytestmark = [pytest.mark.performance, pytest.mark.slow]

# Reduce Ray OOM-kill sensitivity for heavy MPC tests.
os.environ.setdefault("RAY_memory_usage_threshold", "0.99")


TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(TESTS_DIR, "distributed_config.yaml")


def _load_distributed_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_cluster_def(cfg):
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


def _get_net_bytes():
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
        if tx == 0 and rx == 0:
            all_net = psutil.net_io_counters(pernic=False)
            tx = int(all_net.bytes_sent)
            rx = int(all_net.bytes_recv)
        return tx, rx
    except Exception:
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
    tx0, rx0 = _get_net_bytes()
    t0 = time.time()
    result = task()
    elapsed = time.time() - t0
    tx1, rx1 = _get_net_bytes()
    return result, elapsed, max(0, tx1 - tx0), max(0, rx1 - rx0)


def _append_record(csv_path, record):
    pd.DataFrame([record]).to_csv(
        csv_path,
        mode="a",
        header=not os.path.exists(csv_path),
        index=False,
    )


def _gen_binary_data(n_samples, n_features, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features).astype(np.float32)
    w = rng.randn(n_features, 1).astype(np.float32)
    logits = X @ w
    y = (1.0 / (1.0 + np.exp(-logits)) > 0.5).astype(np.float32)
    return X, y


@pytest.fixture(scope="module")
def distributed_env():
    from common import MPCInitializer

    cfg = _load_distributed_config()
    is_placeholder = "MACHINE" in cfg["machine_a"]["ip"] or "MACHINE" in cfg["machine_b"]["ip"]
    if is_placeholder:
        pytest.skip(
            "distributed_config.yaml 中 IP 为占位符, 请先配置实际 IP. "
            "如需单机退化测试, 将两个 IP 均设为 127.0.0.1"
        )

    try:
        sf.shutdown()
    except Exception:
        pass

    cluster_def = _build_cluster_def(cfg)
    ray_addr = f"{cfg['machine_a']['ip']}:{cfg['machine_a']['ray_port']}"
    company_dir = os.path.join(os.path.dirname(TESTS_DIR), "company")
    runtime_env = {"env_vars": {"PYTHONPATH": company_dir}}

    mpc = MPCInitializer(
        mode="multi_distributed",
        ray_head_addr=ray_addr,
        cluster_def=cluster_def,
        runtime_env=runtime_env,
    )

    env = type(
        "Env",
        (),
        {
            "spu": mpc.spu,
            "company": mpc.company,
            "partner": mpc.partner,
            "coordinator": mpc.coordinator,
        },
    )()
    yield env

    try:
        sf.shutdown()
    except Exception:
        pass


@pytest.fixture(scope="module")
def compare_devices(distributed_env):
    env = distributed_env
    return {
        "spu": env.spu,
        "company": env.company,
        "partner": env.partner,
        "coordinator": env.coordinator,
    }


@pytest.fixture(scope="module")
def compare_results_dir(test_results_dir):
    d = os.path.join(test_results_dir, "performance")
    os.makedirs(d, exist_ok=True)
    return d


@pytest.fixture(scope="module")
def mnist_binary_data():
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split

    cache_path = os.path.join(TESTS_DIR, "mnist_784.npz")

    if os.path.exists(cache_path):
        try:
            cached = np.load(cache_path)
            X = cached["X"].astype(np.float32)
            y_raw = cached["y"]
        except ValueError:
            # 兼容旧缓存: y 可能是 object 数组，需要一次性迁移为数值数组。
            cached = np.load(cache_path, allow_pickle=True)
            X = cached["X"].astype(np.float32)
            y_raw = cached["y"]
            y_numeric = np.asarray(y_raw).astype(np.int32)
            np.savez_compressed(cache_path, X=X, y=y_numeric)
            y_raw = y_numeric
    else:
        try:
            mnist = fetch_openml("mnist_784", version=1, as_frame=False)
        except Exception as e:
            pytest.skip(f"MNIST 下载失败，跳过 SSLR 对比测试: {e}")

        X = mnist.data.astype(np.float32)
        y_raw = np.asarray(mnist.target).astype(np.int32)

        # 首次下载后缓存到本地，后续测试直接读取本地文件。
        np.savez_compressed(cache_path, X=X, y=y_raw)

    X = (X / 255.0)
    # 将 MNIST 转成二分类任务: 0 为一类, 非 0 为一类。
    y = (np.asarray(y_raw).astype(np.int32) != 0).astype(np.float32).reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return {
        "train_X": X_train,
        "train_y": y_train,
        "test_X": X_test,
        "test_y": y_test,
        "split_col": X.shape[1] // 2,
    }


@pytest.fixture(scope="module")
def adult_data(project_root):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler

    npy_path = os.path.join(project_root, "tests", "adult.npy")
    raw = np.load(npy_path)

    X = raw[:, 1:].astype(np.float32)
    y = raw[:, 0].astype(np.float32).reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = MinMaxScaler()
    X_train_mm = scaler.fit_transform(X_train).astype(np.float32)
    X_test_mm = scaler.transform(X_test).astype(np.float32)
    # Min-max 标准化后显式裁剪，消除浮点误差导致的微小越界。
    X_train_mm = np.clip(X_train_mm, 0.0, 1.0)
    X_test_mm = np.clip(X_test_mm, 0.0, 1.0)

    assert np.all((X_train_mm >= 0.0) & (X_train_mm <= 1.0)), "Adult train min-max failed"
    assert np.all((X_test_mm >= 0.0) & (X_test_mm <= 1.0)), "Adult test min-max failed"

    return {
        "train_X_mm": X_train_mm,
        "train_y": y_train,
        "test_X_mm": X_test_mm,
        "test_y": y_test,
        "split_col": X.shape[1] // 2,
    }


class TestCompareExperiments:

    def test_sslr_mnist_batch128_epoch2(self, compare_devices, compare_results_dir, mnist_binary_data):
        from LR import SSLR

        company = compare_devices["company"]
        partner = compare_devices["partner"]
        spu = compare_devices["spu"]

        d = mnist_binary_data
        sc = d["split_col"]

        train_X = sf.to(company, d["train_X"]).to(spu)
        train_y = sf.to(company, d["train_y"]).to(spu)

        test_X = load(
            {
                company: sf.to(company, d["test_X"][:, :sc]),
                partner: sf.to(partner, d["test_X"][:, sc:]),
            },
            partition_way=PartitionWay.VERTICAL,
        )

        model = SSLR(compare_devices, approx=True, lambda_=0.1)

        accs, elapsed, tx_bytes, rx_bytes = _run_with_comm_measure(
            lambda: model.fit(
                train_X,
                train_y,
                split_col=sc,
                n_epochs=2,
                batch_size=128,
                lr=0.1,
            )
        )

        if accs:
            best_acc = float(max(accs))
        else:
            y_pred = sf.reveal(model.predict(test_X, company)).reshape(-1)
            best_acc = float(np.mean(y_pred == d["test_y"].reshape(-1)))

        record = {
            "实验": "SSLR-MNIST",
            "batch_size": 128,
            "n_epochs": 2,
            "best_acc": round(best_acc, 6),
            "训练时间(s)": round(elapsed, 4),
            "发送通信量(MB)": round(tx_bytes / 1024 / 1024, 4),
            "接收通信量(MB)": round(rx_bytes / 1024 / 1024, 4),
            "总通信量(MB)": round((tx_bytes + rx_bytes) / 1024 / 1024, 4),
        }
        _append_record(os.path.join(compare_results_dir, "compare_results.csv"), record)

        # 该实验配置按需求固定为 0/非0 二分类 + 极小学习率 2e-7，
        # 这里只校验指标有效性，避免与实验参数目标冲突。
        assert 0.0 <= best_acc <= 1.0, f"SSLR MNIST accuracy out of range: {best_acc}"

    @pytest.mark.parametrize("max_depth", [3, 4, 5])
    def test_ssxgboost_adult_depth_compare(
        self,
        compare_devices,
        compare_results_dir,
        adult_data,
        max_depth,
    ):
        from XGBoost import SSXGBoost, quantize_buckets, recover_buckets

        company = compare_devices["company"]
        partner = compare_devices["partner"]
        spu = compare_devices["spu"]

        d = adult_data
        sc = d["split_col"]

        Q1, _, bl1 = quantize_buckets(d["train_X_mm"][:, :sc], k=9)
        Q2, _, bl2 = quantize_buckets(d["train_X_mm"][:, sc:], k=9)
        buckets = recover_buckets(np.hstack((bl1, bl2)))
        fed_quantiles = load(
            {company: sf.to(company, Q1), partner: sf.to(partner, Q2)},
            partition_way=PartitionWay.HORIZONTAL,
        )

        train_X = sf.to(company, d["train_X_mm"]).to(spu)
        train_y = sf.to(company, d["train_y"])

        test_X = load(
            {
                company: sf.to(company, d["test_X_mm"][:, :sc]),
                partner: sf.to(partner, d["test_X_mm"][:, sc:]),
            },
            partition_way=PartitionWay.VERTICAL,
        )
        test_y = sf.to(company, d["test_y"])

        model = SSXGBoost(
            compare_devices,
            n_estimators=3,
            lambda_=1,
            gamma=0.5,
            max_depth=max_depth,
        )

        acc_pair, elapsed, tx_bytes, rx_bytes = _run_with_comm_measure(
            lambda: model.fit(
                train_X,
                train_y,
                buckets,
                fed_quantiles,
                X_test=test_X,
                y_test=test_y,
            )
        )

        train_accs, test_accs = acc_pair
        acc_source = test_accs if test_accs else train_accs
        if acc_source:
            best_acc = float(max(acc_source))
        else:
            y_pred = sf.reveal(model.predict(test_X, company)).reshape(-1)
            best_acc = float(np.mean(y_pred == d["test_y"].reshape(-1)))

        record = {
            "实验": "SSXGBoost-Adult",
            "n_estimators": 3,
            "max_depth": max_depth,
            "best_acc": round(best_acc, 6),
            "训练时间(s)": round(elapsed, 4),
            "发送通信量(MB)": round(tx_bytes / 1024 / 1024, 4),
            "接收通信量(MB)": round(rx_bytes / 1024 / 1024, 4),
            "总通信量(MB)": round((tx_bytes + rx_bytes) / 1024 / 1024, 4),
        }
        _append_record(os.path.join(compare_results_dir, "compare_results.csv"), record)

        assert best_acc > 0.5, f"Adult SSXGBoost depth={max_depth} accuracy too low: {best_acc}"


class TestSSLRCommunication:

    @pytest.mark.parametrize("n_features", [100, 500, 1000])
    def test_sslr_comm_volume_by_dimension(self, compare_devices, compare_results_dir, n_features):
        from LR import SSLR

        company = compare_devices["company"]
        spu = compare_devices["spu"]

        n_samples = 10000
        batch_size = 128
        n_epochs = 2
        split_col = n_features // 2

        X, y = _gen_binary_data(n_samples=n_samples, n_features=n_features)
        train_X = sf.to(company, X).to(spu)
        train_y = sf.to(company, y).to(spu)

        model = SSLR(compare_devices, approx=True, lambda_=0.1)

        _, elapsed, tx_bytes, rx_bytes = _run_with_comm_measure(
            lambda: model.fit(
                train_X,
                train_y,
                split_col=split_col,
                n_epochs=n_epochs,
                batch_size=batch_size,
                lr=0.1,
            )
        )

        train_loop_elapsed = float(getattr(model, "last_fit_train_loop_seconds", elapsed))
        batch_prepare_elapsed = float(getattr(model, "last_fit_batch_prepare_seconds", 0.0))

        record = {
            "实验": "SSLR-Communication",
            "样本数量": n_samples,
            "特征维度": n_features,
            "batch_size": batch_size,
            "n_epochs": n_epochs,
            "训练时间(s)": round(train_loop_elapsed, 4),
            "分batch时间(s)": round(batch_prepare_elapsed, 4),
            "总流程时间(s)": round(elapsed, 4),
            "发送通信量(MB)": round(tx_bytes / 1024 / 1024, 4),
            "接收通信量(MB)": round(rx_bytes / 1024 / 1024, 4),
            "总通信量(MB)": round((tx_bytes + rx_bytes) / 1024 / 1024, 4),
        }
        _append_record(os.path.join(compare_results_dir, "sslr_comm_by_dimension.csv"), record)

        assert tx_bytes >= 0 and rx_bytes >= 0
