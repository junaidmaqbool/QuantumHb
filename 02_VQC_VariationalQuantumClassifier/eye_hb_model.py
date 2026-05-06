"""
eye_hb_model.py — VQC (Variational Quantum Classifier)
Paper : Farhi & Neven, arXiv 1802.06002 (Google 2018)
Task  : classification (anemic/normal) OR regression (Hb g/dL)
Input : (batch, n_features) float tensor  ← from ResNet-18 + Linear reducer
"""
import torch
import torch.nn as nn

def build_model(n_features: int = 4, n_qubits: int = 4,
                n_layers: int = 2, task: str = "regression"):
    import pennylane as qml

    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def _circuit(inputs, weights):
        # Angle encoding: map each feature to a rotation
        for i in range(n_qubits):
            qml.RY(inputs[..., i], wires=i)
        # Strongly entangling variational layers
        for l in range(n_layers):
            for i in range(n_qubits):
                qml.RX(weights[l, i, 0], wires=i)
                qml.RY(weights[l, i, 1], wires=i)
                qml.RZ(weights[l, i, 2], wires=i)
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            qml.CNOT(wires=[n_qubits - 1, 0])          # ring entanglement
        return qml.expval(qml.PauliZ(0))

    weight_shapes = {"weights": (n_layers, n_qubits, 3)}
    qlayer = qml.qnn.TorchLayer(_circuit, weight_shapes)

    class VQCNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.reducer = nn.Linear(n_features, n_qubits)
            self.bn      = nn.BatchNorm1d(n_qubits)
            self.qlayer  = qlayer

        def forward(self, x):
            x = torch.tanh(self.bn(self.reducer(x))) * torch.pi  # → [−π, π]
            out = self.qlayer(x)                                   # (batch,)
            if task == "classification":
                return torch.sigmoid(out)
            return out   # regression: raw expval ∈ [−1,1]  (de-norm in trainer)

    return VQCNet()
