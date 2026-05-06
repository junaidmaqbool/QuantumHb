# QuantumHb — Quantum Models for Hemoglobin Estimation & Anemia Classification

This repository contains 13 cloned quantum machine learning models, all sourced from
top-tier publications (Nature, Nature Physics, Nature Reviews Physics, PRX Quantum,
Quantum Journal, ACML, Springer QMI). Each model is adapted for the goal of:

- **Hemoglobin (Hb) estimation** — regression on CBC blood count features
- **Anemia classification** — binary (anemic / non-anemic) and multi-class

---

## Repository Structure

```
QuantumHb/
├── 01_QSVM_QuantumEnhancedFeatureSpaces/      ← Nature 2019 | IBM
├── 02_VQC_VariationalQuantumClassifier/        ← Google/arXiv 2018
├── 03_QCNN_QuantumConvolutionalNeuralNet/      ← Nature Physics 2019 | Harvard
├── 04_VQA_TorchQuantum_Framework/              ← Nature Reviews Physics 2021 | MIT
├── 05_DataReUploading_UniversalClassifier/     ← Quantum Journal 2020 | U. Barcelona
├── 06_QuantumTransferLearning/                 ← Quantum Journal 2020 | Xanadu
├── 07_QuantumKernelMethods/                    ← PRX Quantum 2021 | Xanadu/IBM
├── 08_QuantumRandomForest/                     ← Springer QMI 2023
├── 09_QBoost_QuantumBoosting/                  ← ACML 2012 | D-Wave/Google
├── 10_CVQNN_ContinuousVariableQNN/             ← Phys Rev Research 2019 | Xanadu
├── 11_BONUS_QSVM_MedicalClassifier/            ← QSVM on medical data
├── 12_BONUS_QCNN_PhasesOfMatter/               ← QCNN phases (Cong et al. impl.)
└── 13_BONUS_VQA_MetaLearning_QNN/              ← Learning-to-learn with QNNs
```

---

## Model Index

### Model 01 — QSVM: Quantum-Enhanced Feature Spaces
| | |
|---|---|
| **Paper** | Havlíček et al., *"Supervised learning with quantum-enhanced feature spaces"* |
| **Venue** | **Nature, Vol. 567 (2019)** — IBM Quantum |
| **Citations** | ~2,100+ |
| **arXiv** | https://arxiv.org/abs/1804.11326 |
| **Framework** | Qiskit |
| **Use for Hb/Anemia** | Maps CBC blood features into quantum Hilbert space for binary anemia classification using quantum kernel SVM. Best first model to implement. |
| **Key files** | `qiskit_machine_learning/algorithms/classifiers/`, `tutorials/02_quantum_kernel.ipynb` |

---

### Model 02 — VQC: Variational Quantum Classifier
| | |
|---|---|
| **Paper** | Farhi & Neven, *"Classification with Quantum Neural Networks on Near Term Processors"* |
| **Venue** | **Google AI / arXiv 2018** |
| **Citations** | ~1,400+ |
| **arXiv** | https://arxiv.org/abs/1802.06002 |
| **Framework** | PennyLane |
| **Use for Hb/Anemia** | Parametric quantum circuit (PQC) classifier. Works directly on tabular CBC data. Also supports regression for continuous Hb value prediction. |
| **Key files** | `variational_quantum_classifier.ipynb` |

---

### Model 03 — QCNN: Quantum Convolutional Neural Network
| | |
|---|---|
| **Paper** | Cong, Choi & Lukin, *"Quantum Convolutional Neural Networks"* |
| **Venue** | **Nature Physics, Vol. 15 (2019)** — Harvard University |
| **Citations** | ~1,200+ |
| **arXiv** | https://arxiv.org/abs/1810.03787 |
| **Framework** | PennyLane + Qiskit |
| **Use for Hb/Anemia** | Efficient O(log N) parameter model. Encodes CBC features via quantum convolution + pooling layers. Excellent for structured blood count vectors. |
| **Key files** | `qcnn/`, `qcnn_tutorial.py` |

---

### Model 04 — VQA: TorchQuantum Framework (MIT)
| | |
|---|---|
| **Paper** | Cerezo et al., *"Variational Quantum Algorithms"* |
| **Venue** | **Nature Reviews Physics, Vol. 3 (2021)** — Los Alamos + Google |
| **Citations** | ~3,500+ (most cited QML paper) |
| **arXiv** | https://arxiv.org/abs/2012.09265 |
| **Framework** | PyTorch + IBM Qiskit |
| **Use for Hb/Anemia** | Full PyTorch-native quantum ML framework. Build custom QNN architectures. Supports deployment to real IBM quantum hardware. The foundation framework for building your custom QuantumHb model. |
| **Key files** | `torchquantum/`, `examples/`, `docs/` |

---

### Model 05 — Data Re-uploading Universal Classifier
| | |
|---|---|
| **Paper** | Pérez-Salinas et al., *"Data re-uploading for a universal quantum classifier"* |
| **Venue** | **Quantum Journal, Vol. 4 (2020)** — University of Barcelona |
| **Citations** | ~800+ |
| **DOI** | https://quantum-journal.org/papers/q-2020-02-06-226/ |
| **arXiv** | https://arxiv.org/abs/1907.02085 |
| **Framework** | Python (custom) |
| **Use for Hb/Anemia** | Proves a **single qubit** can be a universal classifier. Extremely data-efficient — ideal for small CBC datasets. Novel encoding directly maps Hb values as rotation angles. |
| **Key files** | `single_qubit_classifier.py`, `multi_qubit_classifier.py` |

---

### Model 06 — Quantum Transfer Learning
| | |
|---|---|
| **Paper** | Mari et al., *"Transfer learning in hybrid classical-quantum neural networks"* |
| **Venue** | **Quantum Journal, Vol. 4 (2020)** — Xanadu AI |
| **Citations** | ~600+ |
| **DOI** | https://quantum-journal.org/papers/q-2020-10-09-340/ |
| **arXiv** | https://arxiv.org/abs/1912.08278 |
| **Framework** | PennyLane + PyTorch |
| **Use for Hb/Anemia** | Classical (ResNet/feature extractor) + quantum classifier layer. Best approach when training data is limited — leverages classical pre-trained models to extract features, then uses quantum layer for Hb/anemia decision. |
| **Key files** | `c2q_transfer_learning_cifar.ipynb`, `hybrid_classical_quantum.py` |

---

### Model 07 — Quantum Kernel Methods
| | |
|---|---|
| **Paper** | Schuld, *"Supervised quantum machine learning models are kernel methods"* |
| **Venue** | **PRX Quantum (2021)** — Xanadu AI |
| **Citations** | ~700+ |
| **arXiv** | https://arxiv.org/abs/2101.11020 |
| **Framework** | Qiskit |
| **Use for Hb/Anemia** | Trains the quantum kernel itself to fit your CBC data distribution. Quantum Kernel Alignment (QKA) adapts to Hb/MCH/MCHC/MCV feature space automatically. |
| **Key files** | `docs/tutorials/03_quantum_kernel_training.ipynb` |

---

### Model 08 — Quantum Random Forest
| | |
|---|---|
| **Paper** | Srikumar et al., *"A kernel-based quantum random forest for improved classification"* |
| **Venue** | **Quantum Machine Intelligence, Springer (2023)** |
| **Citations** | ~120+ |
| **DOI** | https://link.springer.com/article/10.1007/s42484-023-00131-2 |
| **arXiv** | https://arxiv.org/abs/2210.02355 |
| **Framework** | PennyLane |
| **Use for Hb/Anemia** | Ensemble of QSVM nodes forming a quantum decision forest. Provably better generalization than a single QSVM. Best ensemble method for anemia classification on CBC data. |
| **Key files** | `qrf.py`, `quantum_random_forest.ipynb` |

---

### Model 09 — QBoost: Quantum Adiabatic Boosting
| | |
|---|---|
| **Paper** | Neven, Denchev, Rose & Macready, *"QBoost: Large scale classifier training with adiabatic quantum optimization"* |
| **Venue** | **ACML 2012** — Google + D-Wave |
| **Citations** | ~400+ |
| **Framework** | D-Wave Ocean SDK |
| **Use for Hb/Anemia** | Quantum annealing-based boosting. Formulates anemia classification as a QUBO problem. Runs on D-Wave quantum annealer. Unique approach — complement to all other gate-based methods. |
| **Key files** | `qboost.py`, `demo.ipynb` |

---

### Model 10 — CV-QNN: Continuous-Variable Quantum Neural Network
| | |
|---|---|
| **Paper** | Killoran et al., *"Continuous-variable quantum neural networks"* |
| **Venue** | **Physical Review Research, Vol. 1 (2019)** — Xanadu AI |
| **Citations** | ~700+ |
| **arXiv** | https://arxiv.org/abs/1806.06871 |
| **Framework** | Strawberry Fields (Xanadu) |
| **Use for Hb/Anemia** | Uses continuous quantum variables (not just qubits). Naturally suited to **regression tasks** — predicting exact Hb values (g/dL) as continuous output. Unique architecture not covered by gate-based models. |
| **Key files** | `cv_classifier.py`, `cv_neural_network.py` |

---

### Bonus 11 — QSVM Medical Classifier
| | |
|---|---|
| **Source** | GlazeDonuts/QSVM |
| **Framework** | Qiskit |
| **Use for Hb/Anemia** | QSVM trained on real medical classification data (breast cancer). Most directly adaptable template for CBC-based anemia binary classification. |
| **Key files** | `QSVM_Qiskit.ipynb` |

---

### Bonus 12 — QCNN Phases of Matter
| | |
|---|---|
| **Source** | Jaybsoni/Quantum-Convolutional-Neural-Networks |
| **Paper** | Follows Cong, Choi & Lukin (Nature Physics 2019) |
| **Framework** | PennyLane |
| **Use for Hb/Anemia** | Full custom QCNN implementation with original convolution and pooling layers from the Cong et al. paper. Adapt for blood severity phase classification (normal / mild anemia / severe anemia). |
| **Key files** | `QCNN_Experiments.ipynb` |

---

### Bonus 13 — VQA Meta-Learning with QNNs
| | |
|---|---|
| **Source** | stfnmangini/Learning2learn |
| **Paper** | Verdon et al., *"Learning to learn with quantum neural networks via classical neural networks"* |
| **Framework** | PennyLane + TensorFlow |
| **Use for Hb/Anemia** | Meta-learning approach — trains a classical network to initialize quantum circuit parameters. Dramatically speeds up training of QNNs on small medical datasets. |
| **Key files** | `learning2learn_demo.ipynb` |

---

## Recommended Implementation Order for QuantumHb

| Priority | Model | Task |
|----------|-------|------|
| 1st | `01_QSVM` | Baseline binary classifier — anemic vs non-anemic |
| 2nd | `02_VQC` | Parametric classifier + Hb regression |
| 3rd | `05_DataReUploading` | Single-qubit Hb classifier (data efficient) |
| 4th | `07_QuantumKernelMethods` | Adaptive kernel on CBC features |
| 5th | `03_QCNN` | Structured CBC data encoding |
| 6th | `08_QuantumRandomForest` | Ensemble — best accuracy |
| 7th | `06_QuantumTransferLearning` | Hybrid classical+quantum |
| 8th | `10_CVQNN` | Continuous Hb value regression |
| 9th | `04_VQA_TorchQuantum` | Custom QuantumHb model (new model) |
| 10th | `09_QBoost` | Quantum annealing classifier |

---

## Input Features (CBC Blood Data for Hb/Anemia)

The models should be trained/adapted to use these standard blood count features:

| Feature | Description | Unit |
|---------|-------------|------|
| Hemoglobin (Hb) | Target variable for regression | g/dL |
| Hematocrit (HCT) | % of RBC in blood volume | % |
| RBC Count | Red blood cell count | million/µL |
| MCV | Mean corpuscular volume | fL |
| MCH | Mean corpuscular hemoglobin | pg |
| MCHC | Mean corpuscular Hb concentration | g/dL |
| RDW | Red cell distribution width | % |
| WBC | White blood cell count | 10³/µL |
| Platelets | Platelet count | 10³/µL |

**Classification Labels:**
- Binary: `0 = Non-Anemic`, `1 = Anemic` (Hb < 12 g/dL women, < 13 g/dL men)
- Multi-class: `0 = Normal`, `1 = Mild`, `2 = Moderate`, `3 = Severe`

---

## Dependencies

```bash
# Core quantum frameworks
pip install qiskit qiskit-machine-learning
pip install pennylane pennylane-qiskit
pip install torchquantum
pip install strawberryfields  # For CV-QNN

# Classical ML
pip install scikit-learn numpy pandas matplotlib

# D-Wave (for QBoost)
pip install dwave-ocean-sdk
```

---

## Custom QuantumHb Model (To Be Built)

The new custom model combining the best of all above will be implemented in:
```
QuantumHb_Custom/
├── quantum_hb_estimator.py       ← Hb regression (CV-QNN + VQC hybrid)
├── anemia_classifier.py          ← Binary/multi-class (QSVM + QRF ensemble)
├── feature_encoder.py            ← CBC data → quantum state encoding
├── train.py
└── evaluate.py
```

---

*Project: QuantumHb — Quantum Machine Learning for Hemoglobin Estimation and Anemia Classification*
