"""
eye_hb_model.py — Kernel-Based Quantum Random Forest (QRF)
Paper : Srikumar et al., Quantum Machine Intelligence, Springer (2023)
GitHub: maiyuren/Quantum-Random-Forest
Task  : classification OR regression
Arch  : Ensemble of QSVM-leaf nodes forming a quantum decision forest
"""
import numpy as np

def build_model(n_qubits: int = 4, n_estimators: int = 10, task: str = "regression"):
    try:
        import pennylane as qml
        from sklearn.svm import SVC, SVR
        from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev, diff_method="parameter-shift")
        def _kernel_circuit(x1, x2):
            """Quantum kernel: |<x1|x2>|² via swap test approximation."""
            # Encode x1
            for i in range(n_qubits):
                qml.RY(x1[i], wires=i)
            # Adjoint encoding of x2 (computes inner product)
            qml.adjoint(qml.AngleEmbedding)(x2, wires=range(n_qubits))
            return qml.probs(wires=range(n_qubits))

        def _qkernel(X_a, X_b):
            """Compute full quantum kernel matrix."""
            K = np.zeros((len(X_a), len(X_b)))
            for i, a in enumerate(X_a):
                for j, b in enumerate(X_b):
                    probs = _kernel_circuit(a * np.pi, b * np.pi)
                    K[i, j] = float(probs[0])   # P(|00..0>) = fidelity proxy
            return K

        class QuantumRandomForest(BaseEstimator):
            """Ensemble of QSVMs — each tree uses a random feature subset."""
            def __init__(self):
                self.estimators_ = []
                self.feat_idx_   = []

            def fit(self, X, y):
                rng = np.random.RandomState(42)
                n_feat = X.shape[1]
                sub = max(1, n_feat // 2)          # random subspace
                for _ in range(n_estimators):
                    idx = rng.choice(n_feat, sub, replace=False)
                    Xs  = np.clip(X[:, idx], -1, 1)
                    # quantum kernel on subset
                    Ktr = _qkernel(Xs, Xs)
                    if task == "classification":
                        clf = SVC(kernel="precomputed", probability=True, C=1.0)
                    else:
                        clf = SVR(kernel="precomputed", C=1.0, epsilon=0.1)
                    clf.fit(Ktr, y)
                    self.estimators_.append(clf)
                    self.feat_idx_.append(idx)
                self.X_train_ = X
                return self

            def predict(self, X):
                preds = []
                for clf, idx in zip(self.estimators_, self.feat_idx_):
                    Xs  = np.clip(X[:, idx], -1, 1)
                    Xtr = np.clip(self.X_train_[:, idx], -1, 1)
                    Kte = _qkernel(Xs, Xtr)
                    preds.append(clf.predict(Kte))
                arr = np.array(preds)            # (n_estimators, n_samples)
                if task == "classification":
                    return (arr.mean(0) > 0.5).astype(int)
                return arr.mean(0)

            def predict_proba(self, X):
                probs = []
                for clf, idx in zip(self.estimators_, self.feat_idx_):
                    Xs  = np.clip(X[:, idx], -1, 1)
                    Xtr = np.clip(self.X_train_[:, idx], -1, 1)
                    Kte = _qkernel(Xs, Xtr)
                    probs.append(clf.predict_proba(Kte)[:, 1])
                return np.array(probs).mean(0)

        return None, QuantumRandomForest()

    except ImportError:
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        print("  [QRF] PennyLane not found → classical RF fallback")
        if task == "classification":
            return None, RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        return None, RandomForestRegressor(n_estimators=n_estimators, random_state=42)
