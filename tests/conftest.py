"""
Shared pytest fixtures for AnonymVFL test suite.

Provides:
- sys.path setup so `company/` modules are importable
- SecretFlow single_sim session-scoped initialization
- Device dict fixtures (SPU, PYU, HEU)
- Sample data path fixtures
- Synthetic data generators
- test_results output directory
"""
import os
import sys
import pytest
import numpy as np

# Raise Ray OOM-kill threshold to 99% so HEU init doesn't get killed
# on memory-constrained machines (must be set before Ray/SF starts).
os.environ.setdefault("RAY_memory_usage_threshold", "0.99")

# ---------------------------------------------------------------------------
# Path setup: make company/ importable as top-level modules
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMPANY_DIR = os.path.join(PROJECT_ROOT, "company")
TESTS_DIR = os.path.join(PROJECT_ROOT, "tests")
for _d in (COMPANY_DIR, PROJECT_ROOT, TESTS_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)

# Also set PYTHONPATH so Ray worker processes can find company/ modules.
# Ray workers are separate processes that don't inherit sys.path changes.
_existing = os.environ.get("PYTHONPATH", "")
_needed = COMPANY_DIR + os.pathsep + PROJECT_ROOT
if COMPANY_DIR not in _existing:
    os.environ["PYTHONPATH"] = _needed + (os.pathsep + _existing if _existing else "")

# ---------------------------------------------------------------------------
# Test results output directory
# ---------------------------------------------------------------------------
TEST_RESULTS_DIR = os.path.join(PROJECT_ROOT, "test_results")


@pytest.fixture(scope="session")
def test_results_dir():
    """Return (and create) the test_results/ output directory."""
    os.makedirs(TEST_RESULTS_DIR, exist_ok=True)
    return TEST_RESULTS_DIR


# ---------------------------------------------------------------------------
# Data path fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def project_root():
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def company_train_csv():
    return os.path.join(COMPANY_DIR, "host_train.csv")


@pytest.fixture(scope="session")
def company_test_csv():
    return os.path.join(COMPANY_DIR, "host_test.csv")


@pytest.fixture(scope="session")
def partner_train_csv():
    return os.path.join(PROJECT_ROOT, "partner", "guest_train.csv")


@pytest.fixture(scope="session")
def partner_test_csv():
    return os.path.join(PROJECT_ROOT, "partner", "guest_test.csv")


# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def make_binary_data():
    """Factory: generate synthetic binary classification data.

    Returns a callable(n_samples, n_features, seed) -> (X, y)
    where X is (n, d) float32 and y is (n, 1) int {0, 1}.
    """
    def _make(n_samples=1000, n_features=20, seed=42):
        rng = np.random.RandomState(seed)
        X = rng.randn(n_samples, n_features).astype(np.float32)
        w = rng.randn(n_features, 1).astype(np.float32)
        logits = X @ w
        prob = 1.0 / (1.0 + np.exp(-logits))
        y = (prob > 0.5).astype(np.float32)
        return X, y
    return _make


# ---------------------------------------------------------------------------
# SecretFlow single_sim fixtures (session-scoped, lazy)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def sf_single_sim():
    """Initialize SecretFlow in single_sim mode (all-in-one process).

    Returns the MPCInitializer instance with .spu, .company, .partner,
    .coordinator, .company_heu, .partner_heu attributes.
    """
    import secretflow as sf
    # Shutdown any residual SF state
    try:
        sf.shutdown()
    except Exception:
        pass

    from common import MPCInitializer
    mpc = MPCInitializer(mode="single_sim")
    yield mpc
    # Teardown
    try:
        sf.shutdown()
    except Exception:
        pass


@pytest.fixture(scope="session")
def devices(sf_single_sim):
    """Dict of SecretFlow devices for single_sim mode."""
    mpc = sf_single_sim
    return {
        "spu": mpc.spu,
        "company": mpc.company,
        "partner": mpc.partner,
        "coordinator": mpc.coordinator,
    }


@pytest.fixture(scope="session")
def heu_devices(sf_single_sim):
    """Tuple of (company_heu, partner_heu) for PSI tests."""
    mpc = sf_single_sim
    return (mpc.company_heu, mpc.partner_heu)


@pytest.fixture(scope="session")
def pyu_devices(sf_single_sim):
    """Tuple of (company_pyu, partner_pyu)."""
    mpc = sf_single_sim
    return (mpc.company, mpc.partner)
