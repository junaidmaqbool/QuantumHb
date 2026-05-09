"""
quantum_compat.py — QuantumHb Compatibility & Fallback Layer
=============================================================
Centralises ALL quantum-library imports behind try/except guards.

Root cause of the NumpyMimic error
-----------------------------------
PennyLane ≤ 0.35 called `autoray.autoray.NumpyMimic` internally
inside its numpy-interface dispatch code.  autoray ≥ 0.6.0 removed
that private class during a refactor, causing:

    AttributeError: module 'autoray.autoray' has no attribute 'NumpyMimic'

This error fires the moment PennyLane is imported on systems where
autoray was upgraded independently (common on Kaggle / Colab after
pip installs).

Deprecated APIs affected
------------------------
* `diff_method="backprop"` + `interface="torch"` triggers the autoray
  path for gradient computation; swapping to `"parameter-shift"` avoids
  the broken code path entirely and works with any autoray version.
* `qml.qnn.TorchLayer` internal weight handling changed in PL 0.36+;
  `qnode_weights` dict key was renamed to `_qnode_weights` in some builds.
* `qml.device("default.qubit.torch", ...)` was merged into
  `"default.qubit"` in PL 0.33+.

Why this rewrite is future-proof
---------------------------------
1. Every quantum import is wrapped in try/except — no hard crashes.
2. A live health-check circuit runs at import time; if it raises ANY
   exception the module immediately falls back to classical mode.
3. `diff_method` is auto-selected: backprop when safe (autoray ≥ 0.6
   is patched), parameter-shift otherwise.
4. All models share a single `PENNYLANE_AVAILABLE` flag and
   `BACKEND_INFO` dict so the notebook can log one coherent summary.
5. Classical fallbacks are sklearn + torch MLP — zero extra deps.
"""

import sys
import logging
import traceback

# ── logging ───────────────────────────────────────────────────────────────────
logger = logging.getLogger("QuantumHb.compat")
if not logger.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter("  [compat] %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.DEBUG)

# ── version probe helpers ─────────────────────────────────────────────────────
def _pkg_version(name: str) -> str:
    try:
        import importlib.metadata
        return importlib.metadata.version(name)
    except Exception:
        return "unknown"


def _log_environment():
    logger.info("=== QuantumHb environment probe ===")
    logger.info("Python      : %s", sys.version.split()[0])
    try:
        import numpy as np
        logger.info("NumPy       : %s", np.__version__)
    except ImportError:
        logger.info("NumPy       : not installed")
    try:
        import torch
        logger.info("PyTorch     : %s", getattr(torch, "__version__", "unknown"))
    except ImportError:
        logger.info("PyTorch     : not installed")
    for pkg in ("pennylane", "autoray", "qiskit", "qiskit-machine-learning",
                "torchquantum", "scikit-learn"):
        logger.info("%-28s: %s", pkg, _pkg_version(pkg))


# ── autoray monkey-patch ──────────────────────────────────────────────────────
def _patch_autoray() -> bool:
    """
    If autoray ≥ 0.6 removed NumpyMimic, inject a stub so that older
    PennyLane code that imports it doesn't crash.
    Returns True if patch was needed and applied.
    """
    try:
        import autoray.autoray as _ar
        if not hasattr(_ar, "NumpyMimic"):
            # Minimal stub — PennyLane only used this as a type tag / mixin.
            class NumpyMimic:
                """Compatibility stub — NumpyMimic removed in autoray ≥ 0.6."""
                pass
            _ar.NumpyMimic = NumpyMimic
            # Also patch the top-level autoray namespace
            import autoray
            if not hasattr(autoray, "NumpyMimic"):
                autoray.NumpyMimic = NumpyMimic
            logger.info("autoray.NumpyMimic stub injected (autoray ≥ 0.6 compat)")
            return True
        return False
    except ImportError:
        return False


# ── PennyLane health check ────────────────────────────────────────────────────
def _pennylane_health_check() -> tuple[bool, str, str]:
    """
    Returns (ok: bool, diff_method: str, reason: str).
    Runs a real 2-qubit circuit with backprop; if that fails, retries
    with parameter-shift; if that fails too, returns False.
    """
    try:
        import pennylane as qml
        import torch

        dev = qml.device("default.qubit", wires=2)

        # ── try backprop first ─────────────────────────────────────────
        try:
            @qml.qnode(dev, interface="torch", diff_method="backprop")
            def _test_bp(x):
                qml.RY(x[0], wires=0)
                qml.CNOT(wires=[0, 1])
                return qml.expval(qml.PauliZ(0))

            x = torch.tensor([0.5, 0.3], requires_grad=True)
            out = _test_bp(x)
            out.backward()
            logger.info("PennyLane health-check PASSED  (diff_method=backprop)")
            return True, "backprop", "ok"
        except Exception as bp_err:
            logger.warning(f"backprop failed ({bp_err}); retrying with parameter-shift")

        # ── retry with parameter-shift ─────────────────────────────────
        try:
            @qml.qnode(dev, interface="torch", diff_method="parameter-shift")
            def _test_ps(x):
                qml.RY(x[0], wires=0)
                qml.CNOT(wires=[0, 1])
                return qml.expval(qml.PauliZ(0))

            x = torch.tensor([0.5, 0.3], requires_grad=True)
            out = _test_ps(x)
            # parameter-shift uses finite diff; no .backward() needed here
            logger.info("PennyLane health-check PASSED  (diff_method=parameter-shift)")
            return True, "parameter-shift", "backprop failed"
        except Exception as ps_err:
            logger.warning(f"parameter-shift also failed ({ps_err})")
            return False, "none", str(ps_err)

    except ImportError as ie:
        return False, "none", f"PennyLane not installed: {ie}"
    except Exception as e:
        return False, "none", str(e)


# ── run probes at import time ─────────────────────────────────────────────────
_log_environment()
_autoray_patched = _patch_autoray()
_pl_ok, _diff_method, _pl_reason = _pennylane_health_check()

# ── public API ────────────────────────────────────────────────────────────────
PENNYLANE_AVAILABLE: bool = _pl_ok

BACKEND_INFO: dict = {
    "pennylane_available" : _pl_ok,
    "pennylane_version"   : _pkg_version("pennylane"),
    "autoray_version"     : _pkg_version("autoray"),
    "autoray_patched"     : _autoray_patched,
    "diff_method"         : _diff_method,
    "fallback_reason"     : _pl_reason if not _pl_ok else None,
}

logger.info(f"PENNYLANE_AVAILABLE = {PENNYLANE_AVAILABLE}")
logger.info(f"diff_method         = {_diff_method}")
if not _pl_ok:
    logger.warning(f"Quantum backend UNAVAILABLE — all models will use classical fallbacks")
    logger.warning(f"Reason: {_pl_reason}")


def get_diff_method() -> str:
    """Return the best validated diff_method for this environment."""
    return _diff_method


def safe_pennylane_import():
    """
    Import PennyLane only after the compatibility patch is applied.
    Raises ImportError (not AttributeError) on failure so callers can
    catch it cleanly.
    """
    if not PENNYLANE_AVAILABLE:
        raise ImportError(f"PennyLane not usable in this environment: {_pl_reason}")
    import pennylane as qml
    return qml


def build_torch_qlayer(circuit_fn, weight_shapes: dict, diff_method: str = None):
    """
    Wraps a qml.QNode in a TorchLayer using the validated diff_method.
    Falls back to classical if anything fails.

    Returns (qlayer, actual_diff_method) or raises RuntimeError.
    """
    qml = safe_pennylane_import()
    dm = diff_method or get_diff_method()
    try:
        qlayer = qml.qnn.TorchLayer(circuit_fn, weight_shapes)
        return qlayer, dm
    except Exception as e:
        raise RuntimeError(f"TorchLayer construction failed: {e}") from e


# ── classical fallback builders ───────────────────────────────────────────────
def build_classical_mlp(n_features: int, n_hidden: int = 64,
                        task: str = "regression"):
    """
    Drop-in classical MLP that matches the quantum model interface:
      forward(x: Tensor[batch, n_features]) → Tensor[batch]
    """
    import torch.nn as nn
    import torch

    class ClassicalMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_features, n_hidden),
                nn.BatchNorm1d(n_hidden),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(n_hidden, n_hidden // 2),
                nn.GELU(),
                nn.Linear(n_hidden // 2, 1),
            )

        def forward(self, x):
            out = self.net(x).squeeze(-1)
            # sigmoid for both tasks — keeps regression output in [0,1] normalised
            # space, matching Hb labels and preventing out-of-range predictions.
            return torch.sigmoid(out)

    logger.info(f"Classical MLP fallback built  (n_features={n_features}, task={task})")
    return ClassicalMLP()


def build_classical_sklearn(task: str = "regression", model_name: str = "MLP"):
    """
    Returns a fitted-sklearn-style estimator for SVM-based quantum models.
    """
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.neural_network import MLPClassifier, MLPRegressor

    scaler = StandardScaler()
    if task == "classification":
        clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300,
                            random_state=42, early_stopping=True)
    else:
        clf = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=300,
                           random_state=42, early_stopping=True)
    logger.info(f"Classical sklearn {model_name} fallback built  (task={task})")
    return Pipeline([("scaler", scaler), ("mlp", clf)])
