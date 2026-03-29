"""
单元测试 — PSI 密码学原语 (rbcl Ristretto255)

测试内容 (对应 test.tex §5.3 PSI密码学基础):
- 哈希确定性: 相同输入 → 相同哈希
- 标量乘法: k·P 生成有效点
- 密钥交换交换律: k_c·(k_p·P) == k_p·(k_c·P)
- 不同密钥 → 不同结果

不依赖 SecretFlow, 仅使用 rbcl.
"""
import pytest
from hashlib import sha512
from rbcl import (
    crypto_core_ristretto255_from_hash,
    crypto_core_ristretto255_scalar_random,
    crypto_scalarmult_ristretto255,
)

pytestmark = pytest.mark.unit


class TestRistretto255Hash:

    def test_deterministic(self):
        """同一输入总是产生相同的群元素"""
        msg = b"test_key_12345"
        h1 = crypto_core_ristretto255_from_hash(sha512(msg).digest())
        h2 = crypto_core_ristretto255_from_hash(sha512(msg).digest())
        assert h1 == h2

    def test_different_inputs(self):
        """不同输入产生不同的群元素"""
        h1 = crypto_core_ristretto255_from_hash(sha512(b"alice").digest())
        h2 = crypto_core_ristretto255_from_hash(sha512(b"bob").digest())
        assert h1 != h2

    def test_output_length(self):
        """Ristretto255 群元素是 32 字节"""
        h = crypto_core_ristretto255_from_hash(sha512(b"key").digest())
        assert len(h) == 32


class TestScalarMultiply:

    def test_result_is_bytes(self):
        k = crypto_core_ristretto255_scalar_random()
        p = crypto_core_ristretto255_from_hash(sha512(b"point").digest())
        result = crypto_scalarmult_ristretto255(k, p)
        assert isinstance(result, bytes) and len(result) == 32

    def test_different_scalars_different_results(self):
        p = crypto_core_ristretto255_from_hash(sha512(b"point").digest())
        k1 = crypto_core_ristretto255_scalar_random()
        k2 = crypto_core_ristretto255_scalar_random()
        r1 = crypto_scalarmult_ristretto255(k1, p)
        r2 = crypto_scalarmult_ristretto255(k2, p)
        # 两个不同随机标量产生不同结果的概率为 1 - 1/(2^252)
        assert r1 != r2


class TestKeyExchangeCommutativity:
    """验证 k_c·(k_p·P) == k_p·(k_c·P), PSI 协议的安全基础"""

    def test_commutativity_single_point(self):
        p = crypto_core_ristretto255_from_hash(sha512(b"shared_id").digest())
        k_c = crypto_core_ristretto255_scalar_random()
        k_p = crypto_core_ristretto255_scalar_random()

        # Company 先乘, Partner 后乘
        kp_then_kc = crypto_scalarmult_ristretto255(
            k_c, crypto_scalarmult_ristretto255(k_p, p)
        )
        # Partner 先乘, Company 后乘
        kc_then_kp = crypto_scalarmult_ristretto255(
            k_p, crypto_scalarmult_ristretto255(k_c, p)
        )
        assert kp_then_kc == kc_then_kp

    def test_commutativity_multiple_points(self):
        """在多个不同点上验证交换律"""
        k_c = crypto_core_ristretto255_scalar_random()
        k_p = crypto_core_ristretto255_scalar_random()

        for label in [b"id_001", b"id_002", b"id_003"]:
            p = crypto_core_ristretto255_from_hash(sha512(label).digest())
            r1 = crypto_scalarmult_ristretto255(
                k_c, crypto_scalarmult_ristretto255(k_p, p)
            )
            r2 = crypto_scalarmult_ristretto255(
                k_p, crypto_scalarmult_ristretto255(k_c, p)
            )
            assert r1 == r2, f"Commutativity failed for {label}"

    def test_intersection_detection(self):
        """模拟 PSI 检测交集: 公共ID → 双方二次加密结果相同"""
        k_c = crypto_core_ristretto255_scalar_random()
        k_p = crypto_core_ristretto255_scalar_random()

        common_ids = [b"user_1", b"user_2"]
        company_only = [b"user_3"]
        partner_only = [b"user_4"]

        def double_encrypt(k_first, k_second, raw_id):
            p = crypto_core_ristretto255_from_hash(sha512(raw_id).digest())
            return crypto_scalarmult_ristretto255(
                k_second, crypto_scalarmult_ristretto255(k_first, p)
            )

        # Company 路径: k_p(k_c(H(id)))
        company_set = {
            uid: double_encrypt(k_c, k_p, uid)
            for uid in common_ids + company_only
        }
        # Partner 路径: k_c(k_p(H(id)))
        partner_set = {
            uid: double_encrypt(k_p, k_c, uid)
            for uid in common_ids + partner_only
        }

        company_vals = set(company_set.values())
        partner_vals = set(partner_set.values())
        intersection = company_vals & partner_vals

        # 应该恰好找到 2 个公共元素
        assert len(intersection) == 2
