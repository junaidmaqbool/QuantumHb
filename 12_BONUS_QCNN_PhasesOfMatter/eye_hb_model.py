"""
eye_hb_model.py — QCNN Phases of Matter (Bonus)
Source: Jaybsoni/Quantum-Convolutional-Neural-Networks (Cong et al. impl.)
Task  : classification OR regression
Note  : Deeper QCNN using the exact convolution/pooling from Cong et al.
"""
import torch, torch.nn as nn

def build_model(n_features: int = 4, n_qubits: int = 4,
                n_layers: int = 1, task: str = "regression"):
    import pennylane as qml

    dev = qml.device("default.qubit", wires=n_qubits)

    def _cong_conv(params, wires):
        """Convolutional unitary from Cong et al. Table S1."""
        qml.RX(params[0], wires=wires[0]); qml.RX(params[1], wires=wires[1])
        qml.RZ(params[2], wires=wires[0]); qml.RZ(params[3], wires=wires[1])
        qml.CNOT(wires=[wires[0], wires[1]])
        qml.RY(params[4], wires=wires[0]); qml.RY(params[5], wires=wires[1])
        qml.CNOT(wires=[wires[1], wires[0]])
        qml.RY(params[6], wires=wires[0]); qml.RY(params[7], wires=wires[1])
        qml.CNOT(wires=[wires[0], wires[1]])
        qml.RX(params[8], wires=wires[1]); qml.RZ(params[9], wires=wires[1])

    def _cong_pool(params, wires):
        """Pooling unitary from Cong et al. — measures one, rotates other."""
        qml.CRZ(params[0], wires=[wires[0], wires[1]])
        qml.PauliX(wires=wires[0])
        qml.CRX(params[1], wires=[wires[0], wires[1]])

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def _circuit(inputs, conv_w, pool_w):
        for i in range(n_qubits):
            qml.RY(inputs[..., i], wires=i)
        active = list(range(n_qubits))
        c, p = 0, 0
        while len(active) > 1:
            for j in range(0, len(active) - 1, 2):
                _cong_conv(conv_w[c], [active[j], active[j+1]]); c += 1
            new = []
            for j in range(0, len(active) - 1, 2):
                _cong_pool(pool_w[p], [active[j], active[j+1]]); p += 1
                new.append(active[j+1])
            active = new
        return qml.expval(qml.PauliZ(active[0]))

    n_conv = sum(n_qubits >> (s+1) for s in range((n_qubits-1).bit_length()))
    qlayer = qml.qnn.TorchLayer(_circuit, {
        "conv_w": (n_conv, 10), "pool_w": (n_conv, 2)})

    class CongQCNNNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.reducer = nn.Linear(n_features, n_qubits)
            self.bn      = nn.BatchNorm1d(n_qubits)
            self.qlayer  = qlayer
        def forward(self, x):
            x = torch.tanh(self.bn(self.reducer(x))) * torch.pi
            out = self.qlayer(x)
            return torch.sigmoid(out) if task == "classification" else out

    return CongQCNNNet()
