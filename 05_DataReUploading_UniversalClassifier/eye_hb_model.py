"""
eye_hb_model.py — Data Re-Uploading Universal Classifier
Paper : Pérez-Salinas et al., Quantum 4, 226 (2020) | U. Barcelona
GitHub: AdrianPerezSalinas/universal_qlassifier
Task  : classification OR regression
Key   : Data is re-uploaded at every layer — single qubit is universal classifier
"""
import torch, torch.nn as nn

def build_model(n_features: int = 4, n_qubits: int = 4,
                n_layers: int = 4, task: str = "regression"):
    import pennylane as qml

    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def _circuit(inputs, weights, bias):
        """
        Data re-uploading: at each layer, apply U(w*x + b) to EVERY qubit.
        This is the core idea of Pérez-Salinas et al. — the same data
        is 'uploaded' multiple times with different learned scalings.
        """
        for l in range(n_layers):
            for q in range(n_qubits):
                # Scale + shift each feature, re-encode into qubit rotations
                phi = weights[l, q] * inputs + bias[l, q]  # (batch, n_features)
                # Use first 3 features as Rot(phi, theta, omega)
                qml.Rot(phi[..., 0 % n_features],
                        phi[..., 1 % n_features],
                        phi[..., 2 % n_features], wires=q)
            # Entangle after each re-uploading layer
            for q in range(n_qubits - 1):
                qml.CNOT(wires=[q, q + 1])
        return qml.expval(qml.PauliZ(0))

    weight_shapes = {
        "weights": (n_layers, n_qubits, n_features),
        "bias"   : (n_layers, n_qubits, n_features),
    }
    qlayer = qml.qnn.TorchLayer(_circuit, weight_shapes)

    class DataReUploadNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.reducer = nn.Linear(n_features, n_features)  # learned feature scaling
            self.bn      = nn.BatchNorm1d(n_features)
            self.qlayer  = qlayer

        def forward(self, x):
            # Soft normalisation to [−π, π]
            x = torch.tanh(self.bn(self.reducer(x))) * torch.pi
            out = self.qlayer(x)
            return torch.sigmoid(out) if task == "classification" else out

    return DataReUploadNet()
