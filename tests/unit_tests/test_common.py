"""
单元测试 — 公共模块 (company/common.py)

测试内容 (对应 test.tex §5.3 公共模块测试):
- 激活函数: sigmoid, approx_sigmoid
- 损失函数: cross_entropy, mean_square_error
- 损失类: ApproxSigmoidCrossEntropy, SigmoidCrossEntropy, MeanSquare
- 辅助函数: to_int_labels, compute_accuracy, compute_f1_metric, compute_for_metric

集成测试 — MPC 初始化与设备验证 (对应 test.tex §5.4.1):
- MPCInitializer 单例模式
- SPU 设备创建与协议配置
- PYU 设备创建 (company, partner, coordinator)
- HEU 设备创建与加解密验证

纯数学运算测试不依赖 SecretFlow; MPC 初始化测试需要 SecretFlow.
"""
import pytest
import numpy as np
import jax.numpy as jnp

import jax

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

    def test_2d_input(self):
        """sigmoid 应支持二维输入"""
        x = jnp.array([[0.0, 1.0], [-1.0, 2.0]])
        y = sigmoid(x)
        assert y.shape == (2, 2)
        assert jnp.all(y >= 0.0) and jnp.all(y <= 1.0)

    def test_derivative_at_zero(self):
        """sigmoid'(0) = sigmoid(0) * (1 - sigmoid(0)) = 0.25"""
        x = jnp.array(0.0)
        grad_fn = jax.grad(lambda z: sigmoid(z))
        np.testing.assert_allclose(float(grad_fn(x)), 0.25, atol=1e-6)


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

    def test_boundary_negative(self):
        """approx_sigmoid(-0.5) = 0.0 (boundary)"""
        np.testing.assert_allclose(float(approx_sigmoid(jnp.array(-0.5))), 0.0, atol=1e-6)

    def test_boundary_positive(self):
        """approx_sigmoid(0.5) = 1.0 (boundary)"""
        np.testing.assert_allclose(float(approx_sigmoid(jnp.array(0.5))), 1.0, atol=1e-6)

    def test_2d_input(self):
        x = jnp.array([[-1.0, 0.0], [0.25, 1.0]])
        y = approx_sigmoid(x)
        expected = jnp.array([[0.0, 0.5], [0.75, 1.0]])
        np.testing.assert_allclose(np.array(y), np.array(expected), atol=1e-6)


# =====================================================================
# Softmax
# =====================================================================

class TestSoftmax:

    def test_output_sums_to_one(self):
        x = jnp.array([1.0, 2.0, 3.0])
        y = softmax(x)
        np.testing.assert_allclose(float(jnp.sum(y)), 1.0, atol=1e-6)

    def test_output_non_negative(self):
        x = jnp.array([-10.0, 0.0, 10.0])
        y = softmax(x)
        assert jnp.all(y >= 0.0)

    def test_uniform_input(self):
        """相等输入 → 均匀分布"""
        x = jnp.array([1.0, 1.0, 1.0])
        y = softmax(x)
        np.testing.assert_allclose(np.array(y), [1/3, 1/3, 1/3], atol=1e-6)

    def test_dominant_input(self):
        """一个极大值 → 对应概率接近 1"""
        x = jnp.array([100.0, 0.0, 0.0])
        y = softmax(x)
        np.testing.assert_allclose(float(y[0]), 1.0, atol=1e-6)

    def test_2d_input(self):
        """二维输入, 每行独立归一化"""
        x = jnp.array([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]])
        y = softmax(x)
        np.testing.assert_allclose(float(jnp.sum(y[0])), 1.0, atol=1e-6)
        np.testing.assert_allclose(float(jnp.sum(y[1])), 1.0, atol=1e-6)

    def test_numerical_stability(self):
        """大值情况下不应溢出"""
        x = jnp.array([1000.0, 1000.0, 1000.0])
        y = softmax(x)
        assert jnp.all(jnp.isfinite(y))
        np.testing.assert_allclose(np.array(y), [1/3, 1/3, 1/3], atol=1e-6)


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

    def test_multiclass(self):
        """多分类 one-hot 交叉熵"""
        y_true = jnp.array([0.0, 1.0, 0.0])
        y_pred = jnp.array([0.1, 0.8, 0.1])
        loss = cross_entropy(y_true, y_pred)
        expected = -jnp.log(0.8 + 1e-12)
        np.testing.assert_allclose(float(loss), float(expected), atol=1e-5)

    def test_all_zeros_true(self):
        """y_true 全零 → loss = 0"""
        y_true = jnp.array([0.0, 0.0])
        y_pred = jnp.array([0.5, 0.5])
        loss = cross_entropy(y_true, y_pred)
        np.testing.assert_allclose(float(loss), 0.0, atol=1e-6)


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

    def test_single_element(self):
        y_true = jnp.array([3.0])
        y_pred = jnp.array([1.0])
        np.testing.assert_allclose(float(mean_square_error(y_true, y_pred)), 4.0, atol=1e-6)

    def test_symmetry(self):
        """MSE(a, b) == MSE(b, a)"""
        y1 = jnp.array([1.0, 2.0, 3.0])
        y2 = jnp.array([4.0, 5.0, 6.0])
        np.testing.assert_allclose(
            float(mean_square_error(y1, y2)),
            float(mean_square_error(y2, y1)),
            atol=1e-8,
        )


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

    def test_binary_threshold(self):
        """binary: 0.5 rounds to 0, 0.5+eps rounds to 1"""
        logits = np.array([[0.49], [0.51]])
        labels = to_int_labels(logits)
        np.testing.assert_array_equal(labels.flatten(), [0, 1])

    def test_single_sample_binary(self):
        logits = np.array([[0.9]])
        labels = to_int_labels(logits)
        np.testing.assert_array_equal(labels.flatten(), [1])

    def test_single_sample_multiclass(self):
        logits = np.array([[0.1, 0.2, 0.7]])
        labels = to_int_labels(logits)
        np.testing.assert_array_equal(labels.flatten(), [2])

    def test_multiclass_tie_picks_first(self):
        """相等logits时 argmax 返回第一个最大值索引"""
        logits = np.array([[0.5, 0.5, 0.0]])
        labels = to_int_labels(logits)
        assert labels.flatten()[0] == 0


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

    def test_single_sample_correct(self):
        np.testing.assert_allclose(compute_accuracy(np.array([1]), np.array([1])), 1.0)

    def test_single_sample_wrong(self):
        np.testing.assert_allclose(compute_accuracy(np.array([0]), np.array([1])), 0.0)

    def test_2d_input_reshape(self):
        """compute_accuracy 内部 reshape → 支持二维输入"""
        y_true = np.array([[0], [1], [0]])
        y_pred = np.array([[0], [1], [1]])
        np.testing.assert_allclose(compute_accuracy(y_true, y_pred), 2.0 / 3.0, atol=1e-8)


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

    def test_all_pred_positive(self):
        """全部预测为 0 (positive)"""
        # y_true=[0,0,1,1], y_pred=[0,0,0,0]
        # TP=2, FP=2, FN=0
        # precision=2/4=0.5, recall=2/2=1.0, F1=2*(0.5*1.0)/(0.5+1.0)=2/3
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 0, 0])
        f1 = compute_f1_metric(y_true, y_pred)
        np.testing.assert_allclose(f1, 2.0 / 3.0, atol=0.05)

    def test_all_same_class(self):
        """所有样本和预测都是同一类"""
        y_true = np.array([0, 0, 0])
        y_pred = np.array([0, 0, 0])
        f1 = compute_f1_metric(y_true, y_pred)
        np.testing.assert_allclose(f1, 1.0, atol=1e-6)

    def test_2d_input(self):
        """二维输入, 内部 reshape 后应正常"""
        y_true = np.array([[0], [1], [0], [1]])
        y_pred = np.array([[0], [1], [0], [1]])
        f1 = compute_f1_metric(y_true, y_pred)
        np.testing.assert_allclose(f1, 1.0, atol=1e-6)


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

    def test_all_pred_positive(self):
        """全部预测为 0 (positive) → pred_negative 全 False → FN=0, TN=0 → FOR≈0"""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 0, 0])
        for_val = compute_for_metric(y_true, y_pred)
        np.testing.assert_allclose(for_val, 0.0, atol=1e-6)

    def test_mixed(self):
        """混合场景验证"""
        # y_true=[0,0,1,1], y_pred=[0,1,0,1]
        # pred_negative = [F,T,F,T]
        # FN = (y_true==0 & y_pred==1) = 1 (idx 1)
        # TN = (y_true==1 & y_pred==1) = 1 (idx 3)
        # FOR = 1/2 = 0.5
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 1])
        for_val = compute_for_metric(y_true, y_pred)
        np.testing.assert_allclose(for_val, 0.5, atol=1e-6)

    def test_2d_input(self):
        """二维输入"""
        y_true = np.array([[0], [0], [1], [1]])
        y_pred = np.array([[0], [0], [1], [1]])
        for_val = compute_for_metric(y_true, y_pred)
        np.testing.assert_allclose(for_val, 0.0, atol=1e-6)


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

    def test_loss_known_value(self):
        """z=0 → approx_sigmoid(0)=0.5 → CE(y=[1,0], p=[0.5,0.5])"""
        y = jnp.array([1.0, 0.0])
        z = jnp.array([0.0, 0.0])
        loss = ApproxSigmoidCrossEntropy.loss(y, z)
        expected = -jnp.log(0.5 + 1e-12)
        np.testing.assert_allclose(float(loss), float(expected), atol=1e-4)

    def test_grad_zero_at_perfect(self):
        """当 approx_sigmoid(z) == y 时, 梯度为 0"""
        y = jnp.array([0.5])
        z = jnp.array([0.0])  # approx_sigmoid(0) = 0.5
        g = ApproxSigmoidCrossEntropy.grad(y, z)
        np.testing.assert_allclose(float(g[0]), 0.0, atol=1e-6)

    def test_hess_at_saturation(self):
        """当 z 在饱和区 (approx_sigmoid=0 或 1) 时, hess=0"""
        y = jnp.array([1.0])
        z = jnp.array([2.0])  # approx_sigmoid(2)=1 → hess=1*(1-1)=0
        h = ApproxSigmoidCrossEntropy.hess(y, z)
        np.testing.assert_allclose(float(h[0]), 0.0, atol=1e-6)


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

    def test_hess_known_value(self):
        """z=0 → sigmoid(0)=0.5 → hess = 0.5*(1-0.5) = 0.25"""
        y = jnp.array([1.0])
        z = jnp.array([0.0])
        h = SigmoidCrossEntropy.hess(y, z)
        np.testing.assert_allclose(float(h[0]), 0.25, atol=1e-6)

    def test_loss_known_value(self):
        """z=0, y=1 → sigmoid(0)=0.5 → CE = -log(0.5)"""
        y = jnp.array([1.0])
        z = jnp.array([0.0])
        loss = SigmoidCrossEntropy.loss(y, z)
        np.testing.assert_allclose(float(loss), -float(jnp.log(0.5 + 1e-12)), atol=1e-4)

    def test_grad_direction_over(self):
        """When sigmoid(z) > y_true, gradient > 0"""
        y = jnp.array([0.0])
        z = jnp.array([2.0])  # sigmoid(2) ≈ 0.88 > 0
        g = SigmoidCrossEntropy.grad(y, z)
        assert float(g[0]) > 0


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
        """softmax CE 的梯度在 one-hot y 时和为 0"""
        y = jnp.array([1.0, 0.0, 0.0])
        z = jnp.array([1.0, 0.5, 0.2])
        g = SoftmaxCrossEntropy.grad(y, z)
        np.testing.assert_allclose(float(jnp.sum(g)), 0.0, atol=1e-6)

    def test_hess_non_negative(self):
        y = jnp.array([1.0, 0.0, 0.0])
        z = jnp.array([1.0, 0.5, 0.2])
        h = SoftmaxCrossEntropy.hess(y, z)
        assert jnp.all(h >= 0)

    def test_loss_non_negative(self):
        y = jnp.array([0.0, 1.0, 0.0])
        z = jnp.array([0.5, 2.0, 0.5])
        loss = SoftmaxCrossEntropy.loss(y, z)
        assert float(loss) >= 0

    def test_loss_uniform_prediction(self):
        """z 全相同 → softmax 均匀 → loss = -log(1/3)"""
        y = jnp.array([1.0, 0.0, 0.0])
        z = jnp.array([0.0, 0.0, 0.0])
        loss = SoftmaxCrossEntropy.loss(y, z)
        np.testing.assert_allclose(float(loss), -float(jnp.log(1.0/3.0 + 1e-12)), atol=1e-4)


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

    def test_loss_known_value(self):
        """MSE([1,2], [3,4]) = mean((4+4)) = 4.0"""
        y_true = jnp.array([1.0, 2.0])
        y_pred = jnp.array([3.0, 4.0])
        np.testing.assert_allclose(float(MeanSquare.loss(y_true, y_pred)), 4.0, atol=1e-6)

    def test_grad_known_value(self):
        """grad = 2*(y_pred - y_true)/n → 2*([2,2])/2 = [2,2]"""
        y_true = jnp.array([1.0, 2.0])
        y_pred = jnp.array([3.0, 4.0])
        g = MeanSquare.grad(y_true, y_pred)
        np.testing.assert_allclose(np.array(g), [2.0, 2.0], atol=1e-6)

    def test_grad_zero_at_perfect(self):
        y = jnp.array([1.0, 2.0])
        g = MeanSquare.grad(y, y)
        np.testing.assert_allclose(np.array(g), [0.0, 0.0], atol=1e-8)

# =====================================================================
# MPC 初始化与设备验证 (原 test_mpc_init.py, 对应 test.tex §5.4.1)
# =====================================================================


class TestMPCInitializerSingleton:
    """验证 MPCInitializer 的单例行为"""

    pytestmark = pytest.mark.integration

    def test_singleton_returns_same_instance(self, sf_single_sim):
        from common import MPCInitializer
        mpc2 = MPCInitializer()
        assert mpc2 is sf_single_sim

    def test_mode_is_single_sim(self, sf_single_sim):
        assert sf_single_sim.mode == "single_sim"


class TestSPUDevice:
    """验证 SPU 设备"""

    pytestmark = pytest.mark.integration

    def test_spu_exists(self, sf_single_sim):
        assert sf_single_sim.spu is not None

    def test_spu_cluster_def(self, sf_single_sim):
        cluster = sf_single_sim.spu.cluster_def
        assert "runtime_config" in cluster
        assert "nodes" in cluster
        # SEMI2K protocol = 3, FM128 field = 3
        assert cluster["runtime_config"]["protocol"] == 3
        assert cluster["runtime_config"]["field"] == 3

    def test_spu_has_three_parties(self, sf_single_sim):
        nodes = sf_single_sim.spu.cluster_def["nodes"]
        parties = {n["party"] for n in nodes}
        assert parties == {"company", "partner", "coordinator"}


class TestPYUDevices:
    """验证 PYU 设备"""

    pytestmark = pytest.mark.integration

    def test_company_pyu(self, sf_single_sim):
        from secretflow import PYU
        assert isinstance(sf_single_sim.company, PYU)

    def test_partner_pyu(self, sf_single_sim):
        from secretflow import PYU
        assert isinstance(sf_single_sim.partner, PYU)

    def test_coordinator_pyu(self, sf_single_sim):
        import secretflow as sf
        # coordinator 是单独命名的 PYU
        coord = sf.PYU("coordinator")
        assert coord is not None

    def test_pyu_computation(self, sf_single_sim):
        """PYU 设备能够执行简单计算"""
        import secretflow as sf
        result = sf_single_sim.company(lambda: 1 + 1)()
        assert sf.reveal(result) == 2


class TestHEUDevices:
    """验证 HEU (同态加密) 设备"""

    pytestmark = pytest.mark.integration

    def test_company_heu_exists(self, sf_single_sim):
        assert sf_single_sim.company_heu is not None

    def test_partner_heu_exists(self, sf_single_sim):
        assert sf_single_sim.partner_heu is not None

    def test_heu_encrypt_decrypt_roundtrip(self, sf_single_sim):
        """验证 HEU 加密→解密 round-trip"""
        import secretflow as sf
        company = sf_single_sim.company
        partner = sf_single_sim.partner
        company_heu = sf_single_sim.company_heu

        # Company 创建一个数组
        data = company(lambda: np.array([1.0, 2.0, 3.0]))()
        # 用 company_heu 加密 (company 是 sk_keeper)
        encrypted = data.to(company_heu).encrypt()
        # 解密回 company
        decrypted = encrypted.to(company)
        result = sf.reveal(decrypted)
        np.testing.assert_allclose(result, [1.0, 2.0, 3.0], atol=0.1)
