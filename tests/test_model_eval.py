"""
模型效果评估测试 (对应 test.tex §5.6 模型效果评估)

测试内容:
- 在公开数据集（Breast Cancer Wisconsin、California Housing）上评估
- SSLR（近似 / 精确）与 SSXGBoost 的模型效果
- 与 sklearn LogisticRegression / XGBClassifier / XGBRegressor 基线对比
- 计算分类指标: 准确率, 精确率, 召回率, F1, AUC
- 计算回归指标: 均方误差(MSE), R²分数
- 输出指标到 CSV, 绘制对比柱状图与训练曲线

数据预处理:
- 样本按 8:1:1 划分（训练集 : Company独有 : Partner独有）
- 独有样本合并作为验证集
- 特征纵向划分：Company 持有前一半，Partner 持有后一半
- LR 训练前特征标准化，XGBoost 不标准化

结果输出到 test_results/model_eval/ 目录.

注意: 本测试使用 single_sim 模式, 无需分布式环境.
"""
import os
import pytest
import numpy as np
import pandas as pd
import secretflow as sf
from secretflow.data.ndarray import load, PartitionWay

pytestmark = pytest.mark.model_eval


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def eval_results_dir(test_results_dir):
    d = os.path.join(test_results_dir, "model_eval")
    os.makedirs(d, exist_ok=True)
    return d


@pytest.fixture(scope="module")
def breast_cancer_data():
    """加载 Breast Cancer Wisconsin 数据集, 按 8:1:1 划分, 特征纵向分割."""
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    data = load_breast_cancer()
    X = data.data.astype(np.float32)
    y = data.target.astype(np.float32)

    # 8:1:1 划分
    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_co_uniq, X_po_uniq, y_co, y_po = train_test_split(
        X_rest, y_rest, test_size=0.5, random_state=42, stratify=y_rest
    )
    # 验证集 = Company独有 + Partner独有
    X_test = np.vstack([X_co_uniq, X_po_uniq])
    y_test = np.concatenate([y_co, y_po])

    # Company 持有前一半特征, Partner 持有后一半
    split_col = X.shape[1] // 2  # 15

    # 标准化版本 (用于 LR)
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train).astype(np.float32)
    X_test_std = scaler.transform(X_test).astype(np.float32)

    return {
        "train_X": X_train,
        "train_y": y_train.reshape(-1, 1),
        "test_X": X_test,
        "test_y": y_test.reshape(-1, 1),
        "train_X_std": X_train_std,
        "test_X_std": X_test_std,
        "split_col": split_col,
        "name": "Breast Cancer Wisconsin",
    }


@pytest.fixture(scope="module")
def california_housing_data():
    """加载 California Housing 数据集, 按 8:1:1 划分, 特征纵向分割."""
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split

    data = fetch_california_housing()
    X = data.data.astype(np.float32)
    y = data.target.astype(np.float32)

    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_co_uniq, X_po_uniq, y_co, y_po = train_test_split(
        X_rest, y_rest, test_size=0.5, random_state=42
    )
    X_test = np.vstack([X_co_uniq, X_po_uniq])
    y_test = np.concatenate([y_co, y_po])

    split_col = X.shape[1] // 2  # 4

    return {
        "train_X": X_train,
        "train_y": y_train.reshape(-1, 1),
        "test_X": X_test,
        "test_y": y_test.reshape(-1, 1),
        "split_col": split_col,
        "name": "California Housing",
    }


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _compute_classification_metrics(y_true, y_pred, y_prob=None):
    """计算分类指标: 准确率, 精确率, 召回率, F1, AUC."""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    metrics = {
        "准确率": accuracy_score(y_true_flat, y_pred_flat),
        "精确率": precision_score(y_true_flat, y_pred_flat, zero_division=0),
        "召回率": recall_score(y_true_flat, y_pred_flat, zero_division=0),
        "F1分数": f1_score(y_true_flat, y_pred_flat, zero_division=0),
    }
    if y_prob is not None:
        try:
            metrics["AUC"] = roc_auc_score(y_true_flat, y_prob.flatten())
        except ValueError:
            metrics["AUC"] = float("nan")
    else:
        metrics["AUC"] = float("nan")
    return metrics


def _compute_regression_metrics(y_true, y_pred):
    """计算回归指标: MSE, R²."""
    from sklearn.metrics import mean_squared_error, r2_score

    return {
        "MSE": mean_squared_error(y_true.flatten(), y_pred.flatten()),
        "R2": r2_score(y_true.flatten(), y_pred.flatten()),
    }


def _plot_regression_comparison(results, dataset_name, png_path):
    """回归指标对比柱状图 (MSE, R²)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.DataFrame(results)
    methods = df["方法"].tolist()
    colors = plt.cm.Set2.colors

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for i, (col, label) in enumerate([("MSE", "均方误差 (MSE)"), ("R2", "R² 分数")]):
        ax = axes[i]
        vals = [float(df[df["方法"] == m].iloc[0][col]) for m in methods]
        bars = ax.bar(methods, vals,
                      color=[colors[j % len(colors)] for j in range(len(methods))],
                      edgecolor="black", linewidth=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.4f}", ha="center", va="bottom", fontsize=9)
        ax.set_ylabel(label)
        ax.set_title(f"{label} — {dataset_name}")
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(png_path), exist_ok=True)
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Breast Cancer — SSLR evaluation (近似 + 精确)
# ---------------------------------------------------------------------------

class TestBreastCancerSSLR:

    @pytest.mark.slow
    def test_approx_sslr(self, devices, breast_cancer_data, eval_results_dir):
        """近似SSLR: Breast Cancer Wisconsin (approx=True)"""
        from LR import SSLR

        company = devices["company"]
        partner = devices["partner"]
        spu = devices["spu"]
        d = breast_cancer_data
        sc = d["split_col"]

        train_X = sf.to(company, d["train_X_std"]).to(spu)
        train_y = sf.to(company, d["train_y"]).to(spu)

        test_X = load(
            {company: sf.to(company, d["test_X_std"][:, :sc]),
             partner: sf.to(partner, d["test_X_std"][:, sc:])},
            partition_way=PartitionWay.VERTICAL,
        )
        test_y = sf.to(company, d["test_y"])

        model = SSLR(devices, approx=True, lambda_=0.01)
        accs = model.fit(
            train_X, train_y,
            X_test=test_X, y_test=test_y,
            n_epochs=20, batch_size=64, val_steps=5, lr=0.1,
        )

        from plot_utils import plot_training_curve
        plot_training_curve(
            list(range(len(accs))), accs, None,
            ylabel="准确率", title="近似SSLR 训练曲线 (Breast Cancer)",
            png_path=os.path.join(eval_results_dir, "approx_sslr_cancer_curve.png"),
        )

        y_pred = sf.reveal(model.predict(test_X, company))
        metrics = _compute_classification_metrics(d["test_y"], y_pred)
        metrics["方法"] = "近似SSLR"
        metrics["数据集"] = "Breast Cancer Wisconsin"

        csv_path = os.path.join(eval_results_dir, "approx_sslr_cancer_metrics.csv")
        pd.DataFrame([metrics]).to_csv(csv_path, index=False)

        assert metrics["准确率"] > 0.5, f"近似SSLR accuracy too low: {metrics['准确率']}"

    @pytest.mark.slow
    def test_exact_sslr(self, devices, breast_cancer_data, eval_results_dir):
        """精确SSLR: Breast Cancer Wisconsin (approx=False)"""
        from LR import SSLR

        company = devices["company"]
        partner = devices["partner"]
        spu = devices["spu"]
        d = breast_cancer_data
        sc = d["split_col"]

        train_X = sf.to(company, d["train_X_std"]).to(spu)
        train_y = sf.to(company, d["train_y"])

        test_X = load(
            {company: sf.to(company, d["test_X_std"][:, :sc]),
             partner: sf.to(partner, d["test_X_std"][:, sc:])},
            partition_way=PartitionWay.VERTICAL,
        )
        test_y = sf.to(company, d["test_y"])

        model = SSLR(devices, approx=False, lambda_=0.01)
        accs = model.fit(
            train_X, train_y,
            X_test=test_X, y_test=test_y,
            n_epochs=20, batch_size=64, val_steps=5, lr=0.1,
        )

        from plot_utils import plot_training_curve
        plot_training_curve(
            list(range(len(accs))), accs, None,
            ylabel="准确率", title="精确SSLR 训练曲线 (Breast Cancer)",
            png_path=os.path.join(eval_results_dir, "exact_sslr_cancer_curve.png"),
        )

        y_pred = sf.reveal(model.predict(test_X, company))
        metrics = _compute_classification_metrics(d["test_y"], y_pred)
        metrics["方法"] = "精确SSLR"
        metrics["数据集"] = "Breast Cancer Wisconsin"

        csv_path = os.path.join(eval_results_dir, "exact_sslr_cancer_metrics.csv")
        pd.DataFrame([metrics]).to_csv(csv_path, index=False)

        assert metrics["准确率"] > 0.5, f"精确SSLR accuracy too low: {metrics['准确率']}"


# ---------------------------------------------------------------------------
# Breast Cancer — SSXGBoost evaluation
# ---------------------------------------------------------------------------

class TestBreastCancerSSXGBoost:

    @pytest.mark.slow
    def test_ssxgboost_cancer(self, devices, breast_cancer_data, eval_results_dir):
        """SSXGBoost 分类: Breast Cancer Wisconsin"""
        from XGBoost import SSXGBoost, quantize_buckets, recover_buckets

        company = devices["company"]
        partner = devices["partner"]
        spu = devices["spu"]
        d = breast_cancer_data
        sc = d["split_col"]

        # XGBoost 不标准化
        Q1, _, bl1 = quantize_buckets(d["train_X"][:, :sc], k=10)
        Q2, _, bl2 = quantize_buckets(d["train_X"][:, sc:], k=10)
        buckets = recover_buckets(np.hstack((bl1, bl2)))
        FedQuantiles = load(
            {company: sf.to(company, Q1), partner: sf.to(partner, Q2)},
            partition_way=PartitionWay.HORIZONTAL,
        )

        train_X = sf.to(company, d["train_X"].astype(np.float32)).to(spu)
        train_y = sf.to(company, d["train_y"].astype(np.float32))

        test_X = load(
            {company: sf.to(company, d["test_X"][:, :sc].astype(np.float32)),
             partner: sf.to(partner, d["test_X"][:, sc:].astype(np.float32))},
            partition_way=PartitionWay.VERTICAL,
        )
        test_y = sf.to(company, d["test_y"].astype(np.float32))

        model = SSXGBoost(devices, n_estimators=3, lambda_=0.1, max_depth=3)
        train_accs, test_accs = model.fit(
            train_X, train_y, buckets, FedQuantiles,
            X_test=test_X, y_test=test_y,
        )

        from plot_utils import plot_training_curve
        steps = list(range(len(train_accs)))
        plot_training_curve(
            steps, train_accs, test_accs if test_accs else None,
            ylabel="准确率", title="SSXGBoost 训练曲线 (Breast Cancer)",
            png_path=os.path.join(eval_results_dir, "ssxgboost_cancer_curve.png"),
        )

        y_pred = sf.reveal(model.predict(test_X, company))
        metrics = _compute_classification_metrics(d["test_y"], y_pred)
        metrics["方法"] = "SSXGBoost"
        metrics["数据集"] = "Breast Cancer Wisconsin"

        csv_path = os.path.join(eval_results_dir, "ssxgboost_cancer_metrics.csv")
        pd.DataFrame([metrics]).to_csv(csv_path, index=False)

        assert metrics["准确率"] > 0.5, f"SSXGBoost accuracy too low: {metrics['准确率']}"


# ---------------------------------------------------------------------------
# California Housing — SSXGBoost regression evaluation
# ---------------------------------------------------------------------------

class TestCaliforniaHousingSSXGBoost:

    @pytest.mark.slow
    def test_ssxgboost_housing(self, devices, california_housing_data, eval_results_dir):
        """SSXGBoost 回归: California Housing"""
        from XGBoost import SSXGBoost, quantize_buckets, recover_buckets

        company = devices["company"]
        partner = devices["partner"]
        spu = devices["spu"]
        d = california_housing_data
        sc = d["split_col"]

        Q1, _, bl1 = quantize_buckets(d["train_X"][:, :sc], k=10)
        Q2, _, bl2 = quantize_buckets(d["train_X"][:, sc:], k=10)
        buckets = recover_buckets(np.hstack((bl1, bl2)))
        FedQuantiles = load(
            {company: sf.to(company, Q1), partner: sf.to(partner, Q2)},
            partition_way=PartitionWay.HORIZONTAL,
        )

        train_X = sf.to(company, d["train_X"].astype(np.float32)).to(spu)
        train_y = sf.to(company, d["train_y"].astype(np.float32))

        test_X = load(
            {company: sf.to(company, d["test_X"][:, :sc].astype(np.float32)),
             partner: sf.to(partner, d["test_X"][:, sc:].astype(np.float32))},
            partition_way=PartitionWay.VERTICAL,
        )
        test_y = sf.to(company, d["test_y"].astype(np.float32))

        model = SSXGBoost(
            devices, n_estimators=5, lambda_=0.1, max_depth=3, mission='Regression'
        )
        model.fit(
            train_X, train_y, buckets, FedQuantiles,
            X_test=test_X, y_test=test_y,
        )

        y_pred = sf.reveal(model.predict(test_X, company))
        metrics = _compute_regression_metrics(d["test_y"], y_pred)
        metrics["方法"] = "SSXGBoost"
        metrics["数据集"] = "California Housing"

        csv_path = os.path.join(eval_results_dir, "ssxgboost_housing_metrics.csv")
        pd.DataFrame([metrics]).to_csv(csv_path, index=False)


# ---------------------------------------------------------------------------
# Sklearn baselines
# ---------------------------------------------------------------------------

class TestSklearnBaselines:

    def test_sklearn_lr_cancer(self, breast_cancer_data, eval_results_dir):
        """sklearn LogisticRegression 基线: Breast Cancer"""
        from sklearn.linear_model import LogisticRegression

        d = breast_cancer_data
        lr = LogisticRegression(max_iter=20, C=100.0, random_state=42)
        lr.fit(d["train_X_std"], d["train_y"].ravel())
        y_pred = lr.predict(d["test_X_std"])
        y_prob = lr.predict_proba(d["test_X_std"])[:, 1]

        metrics = _compute_classification_metrics(d["test_y"], y_pred, y_prob)
        metrics["方法"] = "明文逻辑回归"
        metrics["数据集"] = "Breast Cancer Wisconsin"

        csv_path = os.path.join(eval_results_dir, "sklearn_lr_cancer_metrics.csv")
        pd.DataFrame([metrics]).to_csv(csv_path, index=False)

    def test_sklearn_xgb_cancer(self, breast_cancer_data, eval_results_dir):
        """sklearn-compatible XGBoost 分类基线: Breast Cancer"""
        try:
            from xgboost import XGBClassifier
        except ImportError:
            pytest.skip("xgboost package not installed")

        d = breast_cancer_data
        xgb = XGBClassifier(
            n_estimators=3, max_depth=3, learning_rate=1, reg_lambda=0.1,
            eval_metric="logloss", random_state=42,
        )
        xgb.fit(d["train_X"], d["train_y"].ravel())
        y_pred = xgb.predict(d["test_X"])
        y_prob = xgb.predict_proba(d["test_X"])[:, 1]

        metrics = _compute_classification_metrics(d["test_y"], y_pred, y_prob)
        metrics["方法"] = "明文XGBoost"
        metrics["数据集"] = "Breast Cancer Wisconsin"

        csv_path = os.path.join(eval_results_dir, "sklearn_xgb_cancer_metrics.csv")
        pd.DataFrame([metrics]).to_csv(csv_path, index=False)

    def test_sklearn_xgb_housing(self, california_housing_data, eval_results_dir):
        """sklearn-compatible XGBoost 回归基线: California Housing"""
        try:
            from xgboost import XGBRegressor
        except ImportError:
            pytest.skip("xgboost package not installed")

        d = california_housing_data
        xgb = XGBRegressor(
            n_estimators=5, max_depth=3, learning_rate=1, reg_lambda=0.1,
            random_state=42,
        )
        xgb.fit(d["train_X"], d["train_y"].ravel())
        y_pred = xgb.predict(d["test_X"])

        metrics = _compute_regression_metrics(d["test_y"], y_pred)
        metrics["方法"] = "明文XGBoost"
        metrics["数据集"] = "California Housing"

        csv_path = os.path.join(eval_results_dir, "sklearn_xgb_housing_metrics.csv")
        pd.DataFrame([metrics]).to_csv(csv_path, index=False)


# ---------------------------------------------------------------------------
# Model comparison (aggregation + plotting)
# ---------------------------------------------------------------------------

class TestModelComparison:

    @pytest.mark.slow
    def test_cancer_comparison(self, eval_results_dir):
        """汇总 Breast Cancer 各方法指标并绘制对比图"""
        from plot_utils import plot_model_comparison

        csvs = [
            "sklearn_lr_cancer_metrics.csv",
            "approx_sslr_cancer_metrics.csv",
            "exact_sslr_cancer_metrics.csv",
            "sklearn_xgb_cancer_metrics.csv",
            "ssxgboost_cancer_metrics.csv",
        ]
        all_results = []
        for fname in csvs:
            path = os.path.join(eval_results_dir, fname)
            if os.path.exists(path):
                df = pd.read_csv(path)
                all_results.extend(df.to_dict("records"))

        if len(all_results) < 2:
            pytest.skip("Not enough model results for comparison (need >= 2)")

        csv_path = os.path.join(eval_results_dir, "cancer_comparison.csv")
        pd.DataFrame(all_results).to_csv(csv_path, index=False)

        png_path = os.path.join(eval_results_dir, "cancer_comparison.png")
        plot_model_comparison(all_results, "Breast Cancer Wisconsin", png_path)
        assert os.path.exists(png_path)

    @pytest.mark.slow
    def test_housing_comparison(self, eval_results_dir):
        """汇总 California Housing 各方法回归指标并绘制对比图"""
        csvs = [
            "sklearn_xgb_housing_metrics.csv",
            "ssxgboost_housing_metrics.csv",
        ]
        all_results = []
        for fname in csvs:
            path = os.path.join(eval_results_dir, fname)
            if os.path.exists(path):
                df = pd.read_csv(path)
                all_results.extend(df.to_dict("records"))

        if len(all_results) < 2:
            pytest.skip("Not enough model results for comparison (need >= 2)")

        csv_path = os.path.join(eval_results_dir, "housing_comparison.csv")
        pd.DataFrame(all_results).to_csv(csv_path, index=False)

        png_path = os.path.join(eval_results_dir, "housing_comparison.png")
        _plot_regression_comparison(all_results, "California Housing", png_path)
        assert os.path.exists(png_path)
