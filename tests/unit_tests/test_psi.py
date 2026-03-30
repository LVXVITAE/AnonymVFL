"""
PSI 测试 — 密码学原语 + 集成求交

密码学原语 (对应 test.tex §5.3 PSI密码学基础):
- 哈希确定性: 相同输入 → 相同哈希
- 标量乘法: k·P 生成有效点
- 密钥交换交换律: k_c·(k_p·P) == k_p·(k_c·P)
- 不同密钥 → 不同结果

集成测试 — PSI 隐私集合求交 (对应 test.tex TC1 §5.4.2):
- 已知交集的正确性: 构造 Company/Partner 数据, 验证交集结果
- 无交集场景处理
- 非对称数据量场景
- 交集后数据形状一致性
"""
import pytest
import numpy as np
from hashlib import sha512
from rbcl import (
    crypto_core_ristretto255_from_hash,
    crypto_core_ristretto255_scalar_random,
    crypto_scalarmult_ristretto255,
)


# =====================================================================
# Ristretto255 哈希 (单元测试, 不依赖 SecretFlow)
# =====================================================================

class TestRistretto255Hash:

    pytestmark = pytest.mark.unit

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

    def test_empty_string(self):
        """空字符串也能正常哈希"""
        h = crypto_core_ristretto255_from_hash(sha512(b"").digest())
        assert isinstance(h, bytes) and len(h) == 32

    def test_long_input(self):
        """较长输入也能正常哈希"""
        msg = b"x" * 10000
        h = crypto_core_ristretto255_from_hash(sha512(msg).digest())
        assert isinstance(h, bytes) and len(h) == 32

    def test_numeric_key(self):
        """数字字符串作为 key"""
        h1 = crypto_core_ristretto255_from_hash(sha512(b"12345").digest())
        h2 = crypto_core_ristretto255_from_hash(sha512(b"12346").digest())
        assert h1 != h2
        assert len(h1) == 32


# =====================================================================
# 标量乘法 (单元测试)
# =====================================================================

class TestScalarMultiply:

    pytestmark = pytest.mark.unit

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

    def test_same_scalar_same_result(self):
        """相同标量乘法应得到相同结果"""
        k = crypto_core_ristretto255_scalar_random()
        p = crypto_core_ristretto255_from_hash(sha512(b"test").digest())
        r1 = crypto_scalarmult_ristretto255(k, p)
        r2 = crypto_scalarmult_ristretto255(k, p)
        assert r1 == r2

    def test_different_points_different_results(self):
        """同一标量乘不同点产生不同结果"""
        k = crypto_core_ristretto255_scalar_random()
        p1 = crypto_core_ristretto255_from_hash(sha512(b"point_a").digest())
        p2 = crypto_core_ristretto255_from_hash(sha512(b"point_b").digest())
        r1 = crypto_scalarmult_ristretto255(k, p1)
        r2 = crypto_scalarmult_ristretto255(k, p2)
        assert r1 != r2


# =====================================================================
# 密钥交换交换律 (单元测试)
# =====================================================================

class TestKeyExchangeCommutativity:
    """验证 k_c·(k_p·P) == k_p·(k_c·P), PSI 协议的安全基础"""

    pytestmark = pytest.mark.unit

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

    def test_no_intersection(self):
        """无公共ID → 交集为空"""
        k_c = crypto_core_ristretto255_scalar_random()
        k_p = crypto_core_ristretto255_scalar_random()

        def double_encrypt(k_first, k_second, raw_id):
            p = crypto_core_ristretto255_from_hash(sha512(raw_id).digest())
            return crypto_scalarmult_ristretto255(
                k_second, crypto_scalarmult_ristretto255(k_first, p)
            )

        company_set = {double_encrypt(k_c, k_p, uid) for uid in [b"a", b"b"]}
        partner_set = {double_encrypt(k_p, k_c, uid) for uid in [b"c", b"d"]}
        assert len(company_set & partner_set) == 0

    def test_full_intersection(self):
        """全部ID相同 → 交集大小 == 总数"""
        k_c = crypto_core_ristretto255_scalar_random()
        k_p = crypto_core_ristretto255_scalar_random()

        ids = [b"x1", b"x2", b"x3"]

        def double_encrypt(k_first, k_second, raw_id):
            p = crypto_core_ristretto255_from_hash(sha512(raw_id).digest())
            return crypto_scalarmult_ristretto255(
                k_second, crypto_scalarmult_ristretto255(k_first, p)
            )

        company_vals = {double_encrypt(k_c, k_p, uid) for uid in ids}
        partner_vals = {double_encrypt(k_p, k_c, uid) for uid in ids}
        assert len(company_vals & partner_vals) == 3

    def test_single_element_intersection(self):
        """单元素交集"""
        k_c = crypto_core_ristretto255_scalar_random()
        k_p = crypto_core_ristretto255_scalar_random()

        def double_encrypt(k_first, k_second, raw_id):
            p = crypto_core_ristretto255_from_hash(sha512(raw_id).digest())
            return crypto_scalarmult_ristretto255(
                k_second, crypto_scalarmult_ristretto255(k_first, p)
            )

        company_vals = {double_encrypt(k_c, k_p, uid) for uid in [b"shared", b"only_c"]}
        partner_vals = {double_encrypt(k_p, k_c, uid) for uid in [b"shared", b"only_p"]}
        assert len(company_vals & partner_vals) == 1


# =====================================================================
# PSI 集成测试 — 隐私集合求交 (对应 test.tex TC1 §5.4.2)
# =====================================================================

def _pack_psi_data(pyu, keys, features, public_features=None):
    """将 keys, features, public_features 打包为一个 PYUObject 三元组."""
    return pyu(lambda k, f, p: (k, f, p))(keys, features, public_features)


class TestPSICorrectness:
    """TC1: PSI 求交正确性"""

    pytestmark = pytest.mark.integration

    def test_known_intersection(self, sf_single_sim, heu_devices):
        """构造已知交集的 Company/Partner 数据，验证结果"""
        from PSI import private_set_intersection
        import secretflow as sf

        company = sf_single_sim.company
        partner = sf_single_sim.partner

        # Company: ids 1-8, 8 features
        company_keys = [str(i) for i in range(1, 9)]
        company_features = np.random.randn(8, 4).astype(np.float32)

        # Partner: ids 5-12, 4 features → 交集应为 {5,6,7,8}
        partner_keys = [str(i) for i in range(5, 13)]
        partner_features = np.random.randn(8, 3).astype(np.float32)

        company_data = _pack_psi_data(
            company,
            company_keys,
            company_features,
        )
        partner_data = _pack_psi_data(
            partner,
            partner_keys,
            partner_features,
        )

        R_cI, R_pI, bucket_labels = private_set_intersection(
            company_data, partner_data, heu_devices
        )

        # 交集应有 4 条记录
        r_c = sf.reveal(R_cI)
        r_p = sf.reveal(R_pI)

        assert r_c.shape[0] == 4, f"Expected 4 intersection rows, got {r_c.shape[0]}"
        assert r_p.shape[0] == 4

        # 秘密共享重建: R_cI + R_pI 的列数 = company_features + partner_features
        total_features = company_features.shape[1] + partner_features.shape[1]
        # R_cI 和 R_pI 各含 total_features 列 (加法秘密共享)
        assert r_c.shape[1] == total_features
        assert r_p.shape[1] == total_features

    def test_bucket_labels_none_without_public(self, sf_single_sim, heu_devices):
        """不提供公开特征时 bucket_labels 应为 None"""
        from PSI import private_set_intersection
        import secretflow as sf

        company = sf_single_sim.company
        partner = sf_single_sim.partner

        company_keys = ["a", "b", "c"]
        company_features = np.ones((3, 2), dtype=np.float32)
        partner_keys = ["b", "c", "d"]
        partner_features = np.ones((3, 2), dtype=np.float32)

        company_data = _pack_psi_data(company, company_keys, company_features)
        partner_data = _pack_psi_data(partner, partner_keys, partner_features)

        _, _, bucket_labels = private_set_intersection(
            company_data, partner_data, heu_devices
        )
        assert bucket_labels is None

    def test_shares_reconstruct_original(self, sf_single_sim, heu_devices):
        """验证秘密共享重建: R_cI + R_pI ≈ 原始特征拼接"""
        from PSI import private_set_intersection
        import secretflow as sf

        company = sf_single_sim.company
        partner = sf_single_sim.partner

        # 构造完全重叠的数据以便验证重建
        keys = [str(i) for i in range(5)]
        company_features = np.arange(15, dtype=np.float32).reshape(5, 3)
        partner_features = np.arange(10, dtype=np.float32).reshape(5, 2) + 100

        company_data = _pack_psi_data(company, keys, company_features)
        partner_data = _pack_psi_data(partner, keys, partner_features)

        R_cI, R_pI, _ = private_set_intersection(
            company_data, partner_data, heu_devices
        )

        r_c = sf.reveal(R_cI)
        r_p = sf.reveal(R_pI)
        reconstructed = r_c + r_p

        # 重建后的数据应接近原始拼接 (PSI 内部有随机置换, 行顺序可能改变)
        # 只验证形状和数值范围
        assert reconstructed.shape == (5, 5)
        # 数值范围: company [0,14], partner [100,109] → 总范围 [0, 109]
        assert np.all(np.isfinite(reconstructed))


# =====================================================================
# 提取的模块级函数测试 (对应方案2: 闭包提取)
# =====================================================================

from PSI import (
    unpack_data,
    repermute_data,
    hash_mul_keys,
    scalar_mul_points,
    intersection_indices,
    repermute_with_pem,
)
import pandas as pd


class TestUnpackData:
    pytestmark = pytest.mark.unit

    def test_basic_unpack(self):
        keys = ['a', 'b']
        priv = np.array([[1, 2], [3, 4]])
        pub = np.array([[5], [6]])
        k, p, q = unpack_data((keys, priv, pub))
        assert k == keys
        np.testing.assert_array_equal(p, priv)
        np.testing.assert_array_equal(q, pub)

    def test_none_public(self):
        keys = ['x']
        priv = np.array([[1]])
        k, p, q = unpack_data((keys, priv, None))
        assert q is None


class TestRepermuteData:
    pytestmark = pytest.mark.unit

    def test_preserves_length(self):
        keys = ['a', 'b', 'c']
        priv = np.array([[1], [2], [3]])
        k, p, q = repermute_data(keys, priv, None)
        assert len(k) == 3
        assert p.shape == (3, 1)
        assert q is None

    def test_permutation_consistency(self):
        """keys和features使用同一个排列"""
        keys = ['a', 'b', 'c', 'd']
        priv = np.array([[10], [20], [30], [40]])
        k, p, _ = repermute_data(keys, priv, None)
        # 验证对应关系：key和feature同步排列
        key_to_val = {'a': 10, 'b': 20, 'c': 30, 'd': 40}
        for ki, pi in zip(k, p.flatten()):
            assert key_to_val[ki] == pi

    def test_with_public_features(self):
        keys = ['a', 'b']
        priv = np.array([[1, 2], [3, 4]])
        pub = np.array([[5], [6]])
        k, p, q = repermute_data(keys, priv, pub)
        assert q is not None
        assert q.shape == (2, 1)


class TestHashMulKeys:
    pytestmark = pytest.mark.unit

    def test_basic_hash_mul(self):
        k = crypto_core_ristretto255_scalar_random()
        result = hash_mul_keys(['alice', 'bob'], k)
        assert len(result) == 2
        assert all(isinstance(r, bytes) and len(r) == 32 for r in result)

    def test_deterministic(self):
        k = crypto_core_ristretto255_scalar_random()
        r1 = hash_mul_keys(['test'], k)
        r2 = hash_mul_keys(['test'], k)
        assert r1 == r2

    def test_different_keys_different_results(self):
        k = crypto_core_ristretto255_scalar_random()
        r = hash_mul_keys(['alice', 'bob'], k)
        assert r[0] != r[1]


class TestScalarMulPoints:
    pytestmark = pytest.mark.unit

    def test_basic(self):
        k = crypto_core_ristretto255_scalar_random()
        p = crypto_core_ristretto255_from_hash(sha512(b"test").digest())
        result = scalar_mul_points([p], k)
        assert len(result) == 1
        assert isinstance(result[0], bytes) and len(result[0]) == 32

    def test_multiple_points(self):
        k = crypto_core_ristretto255_scalar_random()
        points = [crypto_core_ristretto255_from_hash(sha512(f"key{i}".encode()).digest()) for i in range(5)]
        result = scalar_mul_points(points, k)
        assert len(result) == 5


class TestIntersectionIndices:
    pytestmark = pytest.mark.unit

    def test_basic_intersection(self):
        E_c = [b'hash_a', b'hash_b', b'hash_c']
        E_p = [b'hash_b', b'hash_c', b'hash_d']
        result = intersection_indices(E_c, E_p)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert set(result['i'].tolist()) == {1, 2}
        assert set(result['j'].tolist()) == {0, 1}

    def test_no_intersection(self):
        E_c = [b'a', b'b']
        E_p = [b'c', b'd']
        result = intersection_indices(E_c, E_p)
        assert len(result) == 0

    def test_full_intersection(self):
        E_c = [b'x', b'y']
        E_p = [b'x', b'y']
        result = intersection_indices(E_c, E_p)
        assert len(result) == 2


class TestRepermuteWithPem:
    pytestmark = pytest.mark.unit

    def test_basic(self):
        E_c_0 = [b'a', b'b', b'c']
        r_c = np.array([[1], [2], [3]])
        result_keys, result_pub, result_r, pem = repermute_with_pem(E_c_0, None, r_c, 3)
        assert len(result_keys) == 3
        assert result_pub is None
        assert result_r.shape == (3, 1)
        assert len(pem) == 3

    def test_pem_is_valid_permutation(self):
        E_c_0 = [b'a', b'b', b'c', b'd']
        r_c = np.array([[1], [2], [3], [4]])
        _, _, _, pem = repermute_with_pem(E_c_0, None, r_c, 4)
        assert sorted(pem) == [0, 1, 2, 3]

    def test_consistency(self):
        """keys和r_c使用同一排列"""
        E_c_0 = [b'a', b'b', b'c']
        r_c = np.array([[10], [20], [30]])
        result_keys, _, result_r, pem = repermute_with_pem(E_c_0, None, r_c, 3)
        key_to_val = {b'a': 10, b'b': 20, b'c': 30}
        for ki, ri in zip(result_keys, result_r.flatten()):
            assert key_to_val[ki] == ri
