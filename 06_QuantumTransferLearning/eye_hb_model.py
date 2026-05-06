"""
eye_hb_model.py — Quantum Transfer Learning
Paper : Mari et al., Quantum 4, 340 (2020) | Xanadu AI
GitHub: XanaduAI/quantum-transfer-learning
Task  : classification OR regression
Arch  : Classical frozen ResNet-18 feature extractor + trainable quantum layer
Note  : The backbone is already handled by the notebook's shared extractor.
        Here we add an additional classical 'dress' layer (Mari et al. Fig 1).
"""
import torch, torch.nn as nn

def build_model(n_features: int = 4, n_qubits: int = 4,
                n_layers: int = 2, task: str = "regression"):
    import pennylane as qml

    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def _dressed_circuit(inputs, pre_w, vqc_w, post_w):
        """
        'Dressed quantum circuit' from Mari et al.:
          classical pre-proc → angle encoding → VQC → classical post-proc
        The classical pre/post layers are thin Linear transforms.
        """
        # Angle-encode pre-processed features
        for i in range(n_qubits):
            qml.RY(inputs[..., i], wires=i)
        # Variational layers
        for l in range(n_layers):
            for i in range(n_qubits):
                qml.Rot(*vqc_w[l, i], wires=i)
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    # Quantum layer returns n_qubits expectation values
    weight_shapes = {"vqc_w": (n_layers, n_qubits, 3)}

    class DressedQNet(nn.Module):
        """Classical-Quantum-Classical 'dressed circuit' (Mari et al.)"""
        def __init__(self):
            super().__init__()
            # Classical 'pre-dress': learned down-projection
            self.pre  = nn.Sequential(nn.Linear(n_features, n_qubits), nn.Tanh())
            # Quantum variational layer (only vqc_w trained here; encoding is data-driven)
            self.vqc_w = nn.Parameter(
                torch.randn(n_layers, n_qubits, 3) * 0.1)
            # Classical 'post-dress': map n_qubits expectation values → 1 output
            self.post = nn.Linear(n_qubits, 1)

        def _q_forward(self, x):
            # x: (batch, n_qubits) after pre-dress
            results = []
            for xi in x:                                # loop over batch
                expvals = _dressed_circuit(
                    xi.unsqueeze(0),
                    None,                                # pre_w unused
                    self.vqc_w,
                    None)                                # post_w unused
                results.append(torch.stack(expvals))
            return torch.stack(results)                  # (batch, n_qubits)

        def forward(self, x):
            x   = self.pre(x) * torch.pi                # (batch, n_qubits)
            q   = self._q_forward(x)                    # (batch, n_qubits)
            out = self.post(q).squeeze(-1)               # (batch,)
            return torch.sigmoid(out) if task == "classification" else out

    return DressedQNet()
