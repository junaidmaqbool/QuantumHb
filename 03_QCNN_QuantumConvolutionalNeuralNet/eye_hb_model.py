"""
eye_hb_model.py — QCNN (Quantum Convolutional Neural Network)
Paper : Cong, Choi & Lukin, Nature Physics 15 (2019)
Task  : classification OR regression
Input : (batch, n_features)
Note  : n_qubits must be a power of 2 (4, 8, 16…)
"""
import torch, torch.nn as nn

def build_model(n_features: int = 4, n_qubits: int = 4,
                n_layers: int = 1, task: str = "regression"):
    import pennylane as qml

    assert n_qubits in (4, 8, 16), "QCNN needs n_qubits ∈ {4,8,16}"
    dev = qml.device("default.qubit", wires=n_qubits)

    def _conv_layer(params, wires):
        """Two-qubit unitary on adjacent pairs (convolutional filter)."""
        qml.RX(params[0], wires=wires[0])
        qml.RX(params[1], wires=wires[1])
        qml.RZ(params[2], wires=wires[0])
        qml.RZ(params[3], wires=wires[1])
        qml.CNOT(wires=[wires[0], wires[1]])
        qml.RY(params[4], wires=wires[0])
        qml.RY(params[5], wires=wires[1])
        qml.CNOT(wires=[wires[1], wires[0]])

    def _pool_layer(params, wires):
        """Pooling: measure one qubit, conditionally rotate the other."""
        qml.CRZ(params[0], wires=[wires[0], wires[1]])
        qml.PauliX(wires=wires[0])
        qml.CRX(params[1], wires=[wires[0], wires[1]])

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def _circuit(inputs, conv_w, pool_w):
        # Encode
        for i in range(n_qubits):
            qml.RY(inputs[..., i % n_qubits], wires=i)
        # Conv + Pool stages (reduces active wires by half each stage)
        active = list(range(n_qubits))
        c_idx, p_idx = 0, 0
        while len(active) > 1:
            # Convolutional pass over all adjacent active pairs
            for j in range(0, len(active) - 1, 2):
                _conv_layer(conv_w[c_idx], [active[j], active[j+1]])
                c_idx += 1
            # Pooling: keep even-indexed wires
            new_active = []
            for j in range(0, len(active) - 1, 2):
                _pool_layer(pool_w[p_idx], [active[j], active[j+1]])
                new_active.append(active[j+1])
                p_idx += 1
            active = new_active
        return qml.expval(qml.PauliZ(active[0]))

    n_stages   = (n_qubits - 1).bit_length()          # log2(n_qubits)
    n_conv_ops = sum(n_qubits >> (s+1) for s in range(n_stages))
    n_pool_ops = n_conv_ops
    wshapes = {
        "conv_w": (n_conv_ops, 8),
        "pool_w": (n_pool_ops, 2),
    }
    qlayer = qml.qnn.TorchLayer(_circuit, wshapes)

    class QCNNNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.reducer = nn.Linear(n_features, n_qubits)
            self.bn      = nn.BatchNorm1d(n_qubits)
            self.qlayer  = qlayer

        def forward(self, x):
            x = torch.tanh(self.bn(self.reducer(x))) * torch.pi
            out = self.qlayer(x)
            return torch.sigmoid(out) if task == "classification" else out

    return QCNNNet()
