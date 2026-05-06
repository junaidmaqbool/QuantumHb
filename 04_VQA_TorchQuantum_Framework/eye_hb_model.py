"""
eye_hb_model.py — TorchQuantum VQA (Variational Quantum Algorithm)
Paper : Cerezo et al., Nature Reviews Physics 3 (2021)  |  MIT Han Lab
Task  : classification OR regression
Input : (batch, n_features)
"""
import torch, torch.nn as nn

def build_model(n_features: int = 4, n_qubits: int = 4,
                n_layers: int = 2, task: str = "regression"):
    try:
        import torchquantum as tq
        import torchquantum.functional as tqf

        class TQVQANet(nn.Module):
            def __init__(self):
                super().__init__()
                self.reducer = nn.Linear(n_features, n_qubits)
                self.bn      = nn.BatchNorm1d(n_qubits)
                # TorchQuantum parameterized gate layers
                self.rx_layers = nn.ModuleList([
                    tq.RX(has_params=True, trainable=True) for _ in range(n_layers * n_qubits)])
                self.ry_layers = nn.ModuleList([
                    tq.RY(has_params=True, trainable=True) for _ in range(n_layers * n_qubits)])
                self.rz_layers = nn.ModuleList([
                    tq.RZ(has_params=True, trainable=True) for _ in range(n_layers * n_qubits)])
                self.measure   = tq.MeasureAll(tq.PauliZ)

            def forward(self, x):
                bsz = x.shape[0]
                x   = torch.tanh(self.bn(self.reducer(x))) * torch.pi
                qdev = tq.QuantumDevice(n_wires=n_qubits, bsz=bsz, device=x.device)
                qdev.reset_states(bsz)
                # Angle encoding
                for i in range(n_qubits):
                    tqf.ry(qdev, wires=i, params=x[:, i].unsqueeze(-1))
                # Variational layers
                idx = 0
                for l in range(n_layers):
                    for i in range(n_qubits):
                        self.rx_layers[idx](qdev, wires=i)
                        self.ry_layers[idx](qdev, wires=i)
                        self.rz_layers[idx](qdev, wires=i)
                        idx += 1
                    for i in range(n_qubits - 1):
                        tqf.cnot(qdev, wires=[i, i+1])
                out = self.measure(qdev)[:, 0]   # expectation of qubit-0 PauliZ
                return torch.sigmoid(out) if task == "classification" else out

        return TQVQANet()

    except ImportError:
        # ── Fallback: pure PennyLane VQA ──────────────────────────────
        import pennylane as qml
        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def _circ(inputs, w):
            for i in range(n_qubits):
                qml.RY(inputs[..., i], wires=i)
            for l in range(n_layers):
                for i in range(n_qubits):
                    qml.Rot(*w[l, i], wires=i)
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i+1])
            return qml.expval(qml.PauliZ(0))

        qlayer = qml.qnn.TorchLayer(_circ, {"w": (n_layers, n_qubits, 3)})

        class FallbackVQA(nn.Module):
            def __init__(self):
                super().__init__()
                self.reducer = nn.Linear(n_features, n_qubits)
                self.bn      = nn.BatchNorm1d(n_qubits)
                self.qlayer  = qlayer
            def forward(self, x):
                x = torch.tanh(self.bn(self.reducer(x))) * torch.pi
                out = self.qlayer(x)
                return torch.sigmoid(out) if task == "classification" else out

        return FallbackVQA()
