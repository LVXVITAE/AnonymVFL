"""
集成测试 — MPC 初始化与设备验证 (对应 test.tex §5.4.1)

测试内容:
- MPCInitializer 单例模式
- SPU 设备创建与协议配置
- PYU 设备创建 (company, partner, coordinator)
- HEU 设备创建与加解密验证
"""
import pytest
import numpy as np

pytestmark = pytest.mark.integration


class TestMPCInitializerSingleton:
    """验证 MPCInitializer 的单例行为"""

    def test_singleton_returns_same_instance(self, sf_single_sim):
        from common import MPCInitializer
        mpc2 = MPCInitializer()
        assert mpc2 is sf_single_sim

    def test_mode_is_single_sim(self, sf_single_sim):
        assert sf_single_sim.mode == "single_sim"


class TestSPUDevice:
    """验证 SPU 设备"""

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
