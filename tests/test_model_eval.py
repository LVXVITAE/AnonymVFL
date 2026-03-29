"""
模型效果评估测试 (对应 test.tex §5.6 模型效果评估)

测试内容:
- 在项目原始 CSV 数据上训练 SSLR / SSXGBoost
- 计算 5 项指标: 准确率, 精确率, 召回率, F1, AUC
- 与 sklearn LogisticRegression / XGBClassifier 基线对比
- 输出指标到 CSV, 绘制对比柱状图与训练曲线

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
def project_data(company_train_csv, company_test_csv, partner_train_csv, partner_test_csv):
    """加载项目原始CSV数据并返回 numpy 数组."""
    company_train = pd.read_csv(company_train_csv)
    company_test = pd.read_csv(company_test_csv)
    partner_train = pd.read_csv(partner_train_csv)
    partner_test = pd.read_csv(partner_test_csv)

    # 取交集 (by id)
    train_merged = pd.merge(company_train, partner_train, on="id", how="inner").sort_values("id")
    test_merged = pd.merge(company_test, partner_test, on="id", how="inner").sort_values("id")

    # Company features: columns between id and Revenue (exclusive)
    company_feat_cols = [c for c in company_train.columns if c not in ("id", "Revenue")]
    partner_feat_cols = [c for c in partner_train.columns if c != "id"]
    split_col = len(company_feat_cols)

    train_X = train_merged[company_feat_cols + partner_feat_cols].to_numpy(dtype=np.float32)
    train_y = train_merged["Revenue"].to_numpy(dtype=np.float32).reshape(-1, 1)
    test_X = test_merged[company_feat_cols + partner_feat_cols].to_numpy(dtype=np.float32)
    test_y = test_merged["Revenue"].to_numpy(dtype=np.float32).reshape(-1, 1)

    return {
        "train_X": train_X,
        "train_y": train_y,
        "test_X": test_X,
        "test_y": test_y,
        "split_col": split_col,
        "n_company_features": split_col,
        "n_partner_features": len(partner_feat_cols),
    }


def _compute_sklearn_metrics(y_true, y_pred, y_prob=None):
    """计算 5 项指标."""
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


# ---------------------------------------------------------------------------
# SSLR evaluation
# ---------------------------------------------------------------------------

class TestSSLREvaluation:

    @pytest.mark.slow
    def test_sslr_on_project_data(self, devices, project_data, eval_results_dir):
        """在项目原始数据上训练 SSLR 并评估"""
        from LR import SSLR

        company = devices["company"]
        partner = devices["partner"]
        spu = devices["spu"]
        d = project_data
        split_col = d["split_col"]

        train_X = sf.to(company, d["train_X"]).to(spu)
        train_y = sf.to(company, d["train_y"]).to(spu)

        test_X = load(
            {company: sf.to(company, d["test_X"][:, :split_col]),
             partner: sf.to(partner, d["test_X"][:, split_col:])},
            partition_way=PartitionWay.VERTICAL,
        )
        test_y = sf.to(company, d["test_y"])

        model = SSLR(devices, approx=True, lambda_=0)
        accs = model.fit(
            train_X, train_y,
            X_test=test_X, y_test=test_y,
            n_epochs=20, batch_size=256, val_steps=5, lr=0.1,
        )

        # 训练曲线
        from plot_utils import plot_training_curve
        plot_training_curve(
            list(range(len(accs))), accs, None,
            ylabel="准确率", title="SSLR 训练曲线 (项目数据)",
            png_path=os.path.join(eval_results_dir, "sslr_training_curve.png"),
        )

        # 评估
        y_pred = sf.reveal(model.predict(test_X, company))
        metrics = _compute_sklearn_metrics(d["test_y"], y_pred)
        metrics["方法"] = "SSLR (VFL)"

        # 保存
        csv_path = os.path.join(eval_results_dir, "sslr_metrics.csv")
        pd.DataFrame([metrics]).to_csv(csv_path, index=False)

        assert metrics["准确率"] > 0.5, f"SSLR accuracy too low: {metrics['准确率']}"


# ---------------------------------------------------------------------------
# SSXGBoost evaluation
# ---------------------------------------------------------------------------

class TestSSXGBoostEvaluation:

    @pytest.mark.slow
    def test_xgboost_on_project_data(self, devices, project_data, eval_results_dir):
        """在项目原始数据上训练 SSXGBoost 并评估"""
        from XGBoost import SSXGBoost, quantize_buckets, recover_buckets

        company = devices["company"]
        partner = devices["partner"]
        spu = devices["spu"]
        d = project_data
        split_col = d["split_col"]

        Q1, _, bl1 = quantize_buckets(d["train_X"][:, :split_col], k=10)
        Q2, _, bl2 = quantize_buckets(d["train_X"][:, split_col:], k=10)
        buckets = recover_buckets(np.hstack((bl1, bl2)))
        FedQuantiles = load(
            {company: sf.to(company, Q1), partner: sf.to(partner, Q2)},
            partition_way=PartitionWay.HORIZONTAL,
        )

        train_X = sf.to(company, d["train_X"].astype(np.float32)).to(spu)
        train_y = sf.to(company, d["train_y"].astype(np.float32))

        test_X = load(
            {company: sf.to(company, d["test_X"][:, :split_col].astype(np.float32)),
             partner: sf.to(partner, d["test_X"][:, split_col:].astype(np.float32))},
            partition_way=PartitionWay.VERTICAL,
        )
        test_y = sf.to(company, d["test_y"].astype(np.float32))

        model = SSXGBoost(
            devices, n_estimators=5, lambda_=1e-4, max_depth=3
        )
        train_accs, test_accs = model.fit(
            train_X, train_y, buckets, FedQuantiles,
            X_test=test_X, y_test=test_y,
        )

        # 训练曲线
        from plot_utils import plot_training_curve
        steps = list(range(len(train_accs)))
        plot_training_curve(
            steps, train_accs, test_accs if test_accs else None,
            ylabel="准确率", title="SSXGBoost 训练曲线 (项目数据)",
            png_path=os.path.join(eval_results_dir, "xgboost_training_curve.png"),
        )

        # 评估
        y_pred = sf.reveal(model.predict(test_X, company))
        metrics = _compute_sklearn_metrics(d["test_y"], y_pred)
        metrics["方法"] = "SSXGBoost (VFL)"

        csv_path = os.path.join(eval_results_dir, "xgboost_metrics.csv")
        pd.DataFrame([metrics]).to_csv(csv_path, index=False)

        assert metrics["准确率"] > 0.5, f"SSXGBoost accuracy too low: {metrics['准确率']}"


# ---------------------------------------------------------------------------
# Sklearn baselines + comparison
# ---------------------------------------------------------------------------

class TestSklearnBaselines:

    def test_sklearn_lr_baseline(self, project_data, eval_results_dir):
        """sklearn LogisticRegression 基线"""
        from sklearn.linear_model import LogisticRegression

        d = project_data
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(d["train_X"], d["train_y"].ravel())
        y_pred = lr.predict(d["test_X"])
        y_prob = lr.predict_proba(d["test_X"])[:, 1]

        metrics = _compute_sklearn_metrics(d["test_y"], y_pred, y_prob)
        metrics["方法"] = "sklearn LR"

        csv_path = os.path.join(eval_results_dir, "sklearn_lr_metrics.csv")
        pd.DataFrame([metrics]).to_csv(csv_path, index=False)

    def test_sklearn_xgb_baseline(self, project_data, eval_results_dir):
        """sklearn-compatible XGBoost 基线 (需 xgboost 包)"""
        try:
            from xgboost import XGBClassifier
        except ImportError:
            pytest.skip("xgboost package not installed")

        d = project_data
        xgb = XGBClassifier(
            n_estimators=50, max_depth=3,
            eval_metric="logloss", random_state=42,
        )
        xgb.fit(d["train_X"], d["train_y"].ravel())
        y_pred = xgb.predict(d["test_X"])
        y_prob = xgb.predict_proba(d["test_X"])[:, 1]

        metrics = _compute_sklearn_metrics(d["test_y"], y_pred, y_prob)
        metrics["方法"] = "XGBoost (sklearn)"

        csv_path = os.path.join(eval_results_dir, "sklearn_xgb_metrics.csv")
        pd.DataFrame([metrics]).to_csv(csv_path, index=False)


class TestModelComparison:

    @pytest.mark.slow
    def test_comparison_plot(self, eval_results_dir):
        """汇总所有指标并绘制对比图"""
        from plot_utils import plot_model_comparison

        csvs = [
            "sslr_metrics.csv",
            "xgboost_metrics.csv",
            "sklearn_lr_metrics.csv",
            "sklearn_xgb_metrics.csv",
        ]
        all_results = []
        for fname in csvs:
            path = os.path.join(eval_results_dir, fname)
            if os.path.exists(path):
                df = pd.read_csv(path)
                all_results.extend(df.to_dict("records"))

        if len(all_results) < 2:
            pytest.skip("Not enough model results for comparison (need >= 2)")

        csv_path = os.path.join(eval_results_dir, "all_model_comparison.csv")
        pd.DataFrame(all_results).to_csv(csv_path, index=False)

        png_path = os.path.join(eval_results_dir, "model_comparison.png")
        plot_model_comparison(all_results, "Shopping Revenue", png_path)
        assert os.path.exists(png_path)
