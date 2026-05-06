"""
eye_hb_model.py — VQA Meta-Learning QNN (Bonus)
Paper : Verdon et al., "Learning to learn with QNNs" (2019)
Source: stfnmangini/Learning2learn (PennyLane + TensorFlow)
Task  : classification OR regression
Arch  : Classical LSTM meta-learner initialises VQC parameters for fast convergence
"""
import torch, torch.nn as nn

def build_model(n_features: int = 4, n_qubits: int = 4,
                n_layers: int = 2, task: str = "regression"):
    import pennylane as qml

    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def _vqc(inputs, weights):
        for i in range(n_qubits):
            qml.RY(inputs[..., i], wires=i)
        for l in range(n_layers):
            for i in range(n_qubits):
                qml.Rot(*weights[l, i], wires=i)
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i+1])
        return qml.expval(qml.PauliZ(0))

    qlayer = qml.qnn.TorchLayer(_vqc, {"weights": (n_layers, n_qubits, 3)})

    class MetaLearnQNN(nn.Module):
        """
        LSTM meta-learner generates smart initial VQC weights.
        In practice the LSTM runs once before training to warm-start VQC params.
        """
        def __init__(self):
            super().__init__()
            self.reducer = nn.Linear(n_features, n_qubits)
            self.bn      = nn.BatchNorm1d(n_qubits)
            # Meta-learner: LSTM that maps task embedding → VQC init weights
            param_dim    = n_layers * n_qubits * 3
            self.meta_lstm = nn.LSTM(n_qubits, param_dim, batch_first=True)
            self.meta_init = None          # set during first forward pass
            self.qlayer  = qlayer

        def _warmstart(self, x_sample):
            """Run LSTM on a small batch to initialise VQC weights."""
            with torch.no_grad():
                seq = x_sample[:8].unsqueeze(0)           # (1, T, n_qubits)
                out, _ = self.meta_lstm(seq)               # (1, T, param_dim)
                init = out[0, -1].view(
                    self.qlayer.qnode_weights["weights"].shape)
                self.qlayer.qnode_weights["weights"].data.copy_(init)

        def forward(self, x):
            x = torch.tanh(self.bn(self.reducer(x))) * torch.pi
            if self.meta_init is None and self.training:
                self._warmstart(x.detach())
                self.meta_init = True
            out = self.qlayer(x)
            return torch.sigmoid(out) if task == "classification" else out

    return MetaLearnQNN()
