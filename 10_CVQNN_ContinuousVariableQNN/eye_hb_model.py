"""
eye_hb_model.py — Continuous-Variable Quantum Neural Network (CV-QNN)
Paper : Killoran et al., Physical Review Research 1 (2019) | Xanadu
GitHub: XanaduAI/quantum-neural-networks
Task  : classification OR regression
Note  : CV-QNN is naturally suited to REGRESSION (continuous output).
        Uses PennyLane gaussian device or default.qubit fallback.
"""
import torch, torch.nn as nn

def build_model(n_features: int = 4, n_qubits: int = 4,
                n_layers: int = 2, task: str = "regression"):
    import pennylane as qml

    # Try gaussian device (true CV), fall back to default.qubit
    try:
        dev = qml.device("strawberryfields.fock", wires=n_qubits, cutoff_dim=5)
        diff_method = "parameter-shift"
    except Exception:
        dev = qml.device("default.qubit", wires=n_qubits)
        diff_method = "backprop"

    @qml.qnode(dev, interface="torch", diff_method=diff_method)
    def _cv_circuit(inputs, weights):
        """
        CV-QNN layer: Interferometer → Squeezing → Interferometer → Displacement
        Approximated here using gate-based rotations for CPU compatibility.
        """
        # Data encoding via displacement / rotation
        for i in range(n_qubits):
            qml.RX(inputs[..., i], wires=i)
        # CV-inspired variational layers
        for l in range(n_layers):
            # Interferometer-like: beam-splitter pairs → CNOT approximation
            for i in range(0, n_qubits - 1, 2):
                qml.CNOT(wires=[i, i + 1])
            for i in range(1, n_qubits - 1, 2):
                qml.CNOT(wires=[i, i + 1])
            # Squeezing-like: single-qubit parameterized rotations
            for i in range(n_qubits):
                qml.RX(weights[l, i, 0], wires=i)
                qml.RY(weights[l, i, 1], wires=i)
            # Displacement-like: bias shifts
            for i in range(n_qubits):
                qml.PhaseShift(weights[l, i, 2], wires=i)
        return qml.expval(qml.PauliZ(0))

    qlayer = qml.qnn.TorchLayer(
        _cv_circuit, {"weights": (n_layers, n_qubits, 3)})

    class CVQNNNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.reducer = nn.Linear(n_features, n_qubits)
            self.bn      = nn.BatchNorm1d(n_qubits)
            self.qlayer  = qlayer
            # CV-QNN naturally outputs continuous values
            # For regression: map expval [−1,1] → Hb range
            self.hb_scale = nn.Parameter(torch.tensor(4.0))   # learnable scale

        def forward(self, x):
            x   = torch.tanh(self.bn(self.reducer(x))) * torch.pi
            out = self.qlayer(x)                    # (batch,) ∈ [−1,1]
            if task == "classification":
                return torch.sigmoid(out)
            # Regression: shift [−1,1] → [0, hb_scale] → meaningful Hb
            return (out + 1.0) / 2.0 * self.hb_scale

    return CVQNNNet()
