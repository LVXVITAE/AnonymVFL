"""
单元测试 — 公共模块 (company/common.py)

测试内容 (对应 test.tex §5.3 公共模块测试):
- 秘密共享拆分 SS_share
- 激活函数: sigmoid, approx_sigmoid, softmax
- 损失函数: cross_entropy, mean_square_error
- 损失类: ApproxSigmoidCrossEntropy, SigmoidCrossEntropy, SoftmaxCrossEntropy, MeanSquare
- 辅助函数: to_int_labels, compute_accuracy, compute_f1_metric, compute_for_metric

所有测试为纯数学运算, 不依赖 SecretFlow.
"""
import pytest
import numpy as np
import jax.numpy as jnp

# 被测模块
from common import (
    sigmoid,
    approx_sigmoid,
    softmax,
    cross_entropy,
    mean_square_error,
    to_int_labels,
    compute_accuracy,
    compute_f1_metric,
    compute_for_metric,
    ApproxSigmoidCrossEntropy,
    SigmoidCrossEntropy,
    SoftmaxCrossEntropy,
    MeanSquare,
)

pytestmark = pytest.mark.unit

# =====================================================================
# Sigmoid
# =====================================================================

class TestSigmoid:

    def test_output_range(self):
        x = jnp.array([-100.0, -1.0, 0.0, 1.0, 100.0])
        y = sigmoid(x)
        assert jnp.all(y >= 0.0) and jnp.all(y <= 1.0)

    def test_symmetry(self):
        """sigmoid(0) == 0.5"""
        np.testing.assert_allclose(float(sigmoid(jnp.array(0.0))), 0.5, atol=1e-6)

    def test_monotonicity(self):
        x = jnp.linspace(-5.0, 5.0, 100)
        y = sigmoid(x)
        diffs = jnp.diff(y)
        assert jnp.all(diffs >= 0)

    def test_known_values(self):
        """sigmoid(large) ≈ 1, sigmoid(-large) ≈ 0"""
        np.testing.assert_allclose(float(sigmoid(jnp.array(50.0))), 1.0, atol=1e-6)
        np.testing.assert_allclose(float(sigmoid(jnp.array(-50.0))), 0.0, atol=1e-6)


# =====================================================================
# Approx Sigmoid (piecewise linear)
# =====================================================================

class TestApproxSigmoid:

    def test_midpoint(self):
        np.testing.assert_allclose(float(approx_sigmoid(jnp.array(0.0))), 0.5, atol=1e-6)

    def test_clip_upper(self):
        """approx_sigmoid(x) for x >= 0.5 should be clipped to 1.0"""
        np.testing.assert_allclose(float(approx_sigmoid(jnp.array(1.0))), 1.0, atol=1e-6)

    def test_clip_lower(self):
        """approx_sigmoid(x) for x <= -0.5 should be clipped to 0.0"""
        np.testing.assert_allclose(float(approx_sigmoid(jnp.array(-1.0))), 0.0, atol=1e-6)

    def test_linear_range(self):
        """Between -0.5 and 0.5, approx_sigmoid(x) = x + 0.5"""
        x = jnp.array(0.25)
        np.testing.assert_allclose(float(approx_sigmoid(x)), 0.75, atol=1e-6)

    def test_batch(self):
        x = jnp.array([-2.0, -0.5, 0.0, 0.5, 2.0])
        y = approx_sigmoid(x)
        expected = jnp.array([0.0, 0.0, 0.5, 1.0, 1.0])
        np.testing.assert_allclose(np.array(y), np.array(expected), atol=1e-6)

# =====================================================================
# Cross Entropy
# =====================================================================

class TestCrossEntropy:

    def test_perfect_prediction(self):
        """交叉熵在完美预测时接近 0"""
        y_true = jnp.array([1.0, 0.0])
        y_pred = jnp.array([1.0 - 1e-12, 1e-12])
        loss = cross_entropy(y_true, y_pred)
        assert float(loss) < 1e-5

    def test_worst_prediction(self):
        """交叉熵在完全错误预测时很大"""
        y_true = jnp.array([1.0, 0.0])
        y_pred = jnp.array([1e-12, 1.0 - 1e-12])
        loss = cross_entropy(y_true, y_pred)
        assert float(loss) > 10.0

    def test_non_negative(self):
        y_true = jnp.array([0.3, 0.7])
        y_pred = jnp.array([0.4, 0.6])
        loss = cross_entropy(y_true, y_pred)
        assert float(loss) >= 0


# =====================================================================
# Mean Square Error
# =====================================================================

class TestMeanSquareError:

    def test_zero_error(self):
        y = jnp.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(float(mean_square_error(y, y)), 0.0, atol=1e-8)

    def test_known_value(self):
        y_true = jnp.array([1.0, 2.0])
        y_pred = jnp.array([3.0, 4.0])
        # MSE = mean((2^2 + 2^2)) = 4.0
        np.testing.assert_allclose(float(mean_square_error(y_true, y_pred)), 4.0, atol=1e-6)

    def test_non_negative(self):
        rng = np.random.RandomState(0)
        y_true = jnp.array(rng.randn(10))
        y_pred = jnp.array(rng.randn(10))
        assert float(mean_square_error(y_true, y_pred)) >= 0


# =====================================================================
# to_int_labels
# =====================================================================

class TestToIntLabels:

    def test_binary(self):
        logits = np.array([[0.3], [0.7], [0.5]])
        labels = to_int_labels(logits)
        np.testing.assert_array_equal(labels.flatten(), [0, 1, 0])

    def test_multiclass(self):
        logits = np.array([[0.1, 0.8, 0.1], [0.9, 0.05, 0.05]])
        labels = to_int_labels(logits)
        np.testing.assert_array_equal(labels.flatten(), [1, 0])


# =====================================================================
# compute_accuracy
# =====================================================================

class TestComputeAccuracy:

    def test_perfect(self):
        y = np.array([0, 1, 0, 1])
        assert compute_accuracy(y, y) == 1.0

    def test_half_correct(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0])
        np.testing.assert_allclose(compute_accuracy(y_true, y_pred), 0.5, atol=1e-8)

    def test_all_wrong(self):
        y_true = np.array([0, 0, 0])
        y_pred = np.array([1, 1, 1])
        np.testing.assert_allclose(compute_accuracy(y_true, y_pred), 0.0, atol=1e-8)


# =====================================================================
# compute_f1_metric
# =====================================================================

class TestComputeF1Metric:
    """注意: 源码中 positive = (y_true == 0), pred_positive = (y_pred == 0)"""

    def test_perfect(self):
        y = np.array([0, 1, 0, 1])
        f1 = compute_f1_metric(y, y)
        np.testing.assert_allclose(f1, 1.0, atol=1e-6)

    def test_all_pred_negative(self):
        """全部预测为 1 (negative), 所有 positive(0) 被漏掉 → TP=0"""
        y_true = np.array([0, 0, 0])
        y_pred = np.array([1, 1, 1])
        f1 = compute_f1_metric(y_true, y_pred)
        assert f1 < 0.01  # TP=0 → F1≈0

    def test_mixed(self):
        # y_true=[0,0,1,1], y_pred=[0,1,0,1]
        # positive_mask: y_true==0 → [T,T,F,F]
        # pred_positive: y_pred==0 → [T,F,T,F]
        # TP = (y_true==0 & y_pred==0) = 1 (idx 0)
        # FP = (y_true==1 & y_pred==0) = 1 (idx 2)
        # FN = (y_true==0 & y_pred==1) = 1 (idx 1)
        # precision=1/2, recall=1/2, F1=0.5
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 1])
        f1 = compute_f1_metric(y_true, y_pred)
        np.testing.assert_allclose(f1, 0.5, atol=0.05)


# =====================================================================
# compute_for_metric
# =====================================================================

class TestComputeForMetric:
    """FOR = FN / (FN + TN), 其中 FN = (y_true==0 & y_pred==1), TN = (y_true==1 & y_pred==1)"""

    def test_perfect(self):
        """完美预测 → FN=0 → FOR=0"""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        for_val = compute_for_metric(y_true, y_pred)
        np.testing.assert_allclose(for_val, 0.0, atol=1e-6)

    def test_all_pred_negative(self):
        """全部预测为 1 (negative)"""
        # y_true=[0,0,1,1], y_pred=[1,1,1,1]
        # FN = (y_true==0 & y_pred==1) = 2
        # TN = (y_true==1 & y_pred==1) = 2
        # FOR = 2/4 = 0.5
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 1, 1])
        for_val = compute_for_metric(y_true, y_pred)
        np.testing.assert_allclose(for_val, 0.5, atol=1e-6)


# =====================================================================
# ApproxSigmoidCrossEntropy — loss, grad, hess
# =====================================================================

class TestApproxSigmoidCrossEntropy:

    def test_loss_shape(self):
        y = jnp.array([1.0, 0.0])
        z = jnp.array([0.0, 0.0])
        loss = ApproxSigmoidCrossEntropy.loss(y, z)
        assert loss.shape == ()

    def test_grad_shape(self):
        y = jnp.array([1.0, 0.0])
        z = jnp.array([0.5, -0.5])
        g = ApproxSigmoidCrossEntropy.grad(y, z)
        assert g.shape == y.shape

    def test_grad_direction(self):
        """When prediction < true → gradient should be negative"""
        y = jnp.array([1.0])
        z = jnp.array([-1.0])  # approx_sigmoid(-1) = 0 < 1
        g = ApproxSigmoidCrossEntropy.grad(y, z)
        assert float(g[0]) < 0  # y_pred - y_true = 0 - 1 = -1

    def test_hess_non_negative(self):
        y = jnp.array([1.0, 0.0])
        z = jnp.array([0.2, -0.3])
        h = ApproxSigmoidCrossEntropy.hess(y, z)
        assert jnp.all(h >= 0)


# =====================================================================
# SigmoidCrossEntropy — loss, grad, hess
# =====================================================================

class TestSigmoidCrossEntropy:

    def test_loss_non_negative(self):
        y = jnp.array([1.0, 0.0])
        z = jnp.array([2.0, -2.0])
        loss = SigmoidCrossEntropy.loss(y, z)
        assert float(loss) >= 0

    def test_grad_shape(self):
        y = jnp.array([1.0, 0.0, 1.0])
        z = jnp.array([0.5, -0.5, 1.0])
        g = SigmoidCrossEntropy.grad(y, z)
        assert g.shape == y.shape

    def test_grad_zero_at_perfect(self):
        """When sigmoid(z) == y, gradient == 0"""
        y = jnp.array([0.5])
        z = jnp.array([0.0])  # sigmoid(0) = 0.5
        g = SigmoidCrossEntropy.grad(y, z)
        np.testing.assert_allclose(float(g[0]), 0.0, atol=1e-6)

    def test_hess_positive(self):
        y = jnp.array([1.0])
        z = jnp.array([0.0])
        h = SigmoidCrossEntropy.hess(y, z)
        assert float(h[0]) > 0  # sigmoid(0)*(1-sigmoid(0)) = 0.25


# =====================================================================
# SoftmaxCrossEntropy — loss, grad, hess
# =====================================================================

class TestSoftmaxCrossEntropy:

    def test_loss_shape(self):
        y = jnp.array([1.0, 0.0, 0.0])
        z = jnp.array([1.0, 0.5, 0.2])
        loss = SoftmaxCrossEntropy.loss(y, z)
        assert loss.shape == ()

    def test_grad_shape(self):
        y = jnp.array([1.0, 0.0, 0.0])
        z = jnp.array([1.0, 0.5, 0.2])
        g = SoftmaxCrossEntropy.grad(y, z)
        assert g.shape == y.shape

    def test_grad_sum_zero(self):
        """Gradient of softmax CE sums to 0 when y is one-hot"""
        y = jnp.array([1.0, 0.0, 0.0])
        z = jnp.array([1.0, 0.5, 0.2])
        g = SoftmaxCrossEntropy.grad(y, z)
        np.testing.assert_allclose(float(jnp.sum(g)), 0.0, atol=1e-6)


# =====================================================================
# MeanSquare — loss, grad, hess
# =====================================================================

class TestMeanSquareLoss:

    def test_loss_zero(self):
        y = jnp.array([1.0, 2.0])
        loss = MeanSquare.loss(y, y)
        np.testing.assert_allclose(float(loss), 0.0, atol=1e-8)

    def test_grad_shape(self):
        y_true = jnp.array([1.0, 2.0])
        y_pred = jnp.array([1.5, 2.5])
        g = MeanSquare.grad(y_true, y_pred)
        assert g.shape == y_true.shape

    def test_grad_direction(self):
        """When y_pred > y_true, gradient should be positive"""
        y_true = jnp.array([0.0])
        y_pred = jnp.array([1.0])
        g = MeanSquare.grad(y_true, y_pred)
        assert float(g[0]) > 0

    def test_hess_constant(self):
        y_true = jnp.array([1.0, 2.0, 3.0])
        y_pred = jnp.array([0.0, 0.0, 0.0])
        h = MeanSquare.hess(y_true, y_pred)
        # hess = 2 / n
        np.testing.assert_allclose(float(h), 2.0 / 3.0, atol=1e-6)
