# Life — Deep Neural Network from Scratch

A lightweight, high-performance 5-layer Deep Neural Network built entirely from scratch using **NumPy** for binary classification on medical diagnostic data (Wisconsin Breast Cancer dataset).

---

## 👤 Author

* **Author:** Mayank Bhatt
* **GitHub:** [@CodeStrikerMayank](https://github.com/CodeStrikerMayank)
* **Email:** [mayankbhatt9tha3@gmail.com](mailto:mayankbhatt9tha3@gmail.com)

---

## 📌 Project Overview

**Life** implements the foundational mathematics of deep learning without relying on high-level frameworks like PyTorch or TensorFlow. All core components — forward propagation, Leaky ReLU and Sigmoid activations, He/Kaiming weight initialization, binary cross-entropy loss, and analytical backpropagation — are implemented directly with vectorized NumPy matrix operations.

### Key Highlights
- **100% From Scratch:** Pure NumPy matrix implementation of forward and backward passes.
- **5-Layer Architecture:** (30 $\to$ 16 $\to$ 8 $\to$ 4 $\to$ 2 $\to$ 1) with Leaky ReLU hidden layers and a Sigmoid output.
- **Data Leakage-Free Pipeline:** Normalization statistics ($\mu, \sigma$) are computed strictly from the training split and applied consistently across validation and test splits.
- **Best-Model Checkpointing:** Tracks validation loss across epochs and saves the optimal model weights and normalization parameters to a compressed `data.npz` archive.
- **Modern Packaging:** Managed with [`uv`](https://github.com/astral-sh/uv) and compliant with `pyproject.toml` standard packaging.

---

## 🧠 Network Architecture

```
Input Features (30)
       │
       ▼
┌──────────────────────────────┐
│  Layer 1: Dense (30 -> 16)   │  Activation: Leaky ReLU (α = 0.01)
└──────────────────────────────┘
       │
       ▼
┌──────────────────────────────┐
│  Layer 2: Dense (16 -> 8)    │  Activation: Leaky ReLU (α = 0.01)
└──────────────────────────────┘
       │
       ▼
┌──────────────────────────────┐
│  Layer 3: Dense (8 -> 4)     │  Activation: Leaky ReLU (α = 0.01)
└──────────────────────────────┘
       │
       ▼
┌──────────────────────────────┐
│  Layer 4: Dense (4 -> 2)     │  Activation: Leaky ReLU (α = 0.01)
└──────────────────────────────┘
       │
       ▼
┌──────────────────────────────┐
│  Layer 5: Output (2 -> 1)    │  Activation: Sigmoid
└──────────────────────────────┘
       │
       ▼
Predicted Probability [0, 1] (0 = Malignant, 1 = Benign)
```

| Layer | Dimensions ($W \times X + b$) | Activation | Initialization |
| :--- | :--- | :--- | :--- |
| **Layer 1** | $W_1 \in \mathbb{R}^{30 \times 16}, b_1 \in \mathbb{R}^{1 \times 16}$ | Leaky ReLU ($\alpha=0.01$) | He Normal: $\mathcal{N}(0, \sqrt{2/30})$ |
| **Layer 2** | $W_2 \in \mathbb{R}^{16 \times 8}, b_2 \in \mathbb{R}^{1 \times 8}$ | Leaky ReLU ($\alpha=0.01$) | He Normal: $\mathcal{N}(0, \sqrt{2/16})$ |
| **Layer 3** | $W_3 \in \mathbb{R}^{8 \times 4}, b_3 \in \mathbb{R}^{1 \times 4}$ | Leaky ReLU ($\alpha=0.01$) | He Normal: $\mathcal{N}(0, \sqrt{2/8})$ |
| **Layer 4** | $W_4 \in \mathbb{R}^{4 \times 2}, b_4 \in \mathbb{R}^{1 \times 2}$ | Leaky ReLU ($\alpha=0.01$) | He Normal: $\mathcal{N}(0, \sqrt{2/4})$ |
| **Layer 5 (Output)** | $W_5 \in \mathbb{R}^{2 \times 1}, b_5 \in \mathbb{R}^{1 \times 1}$ | Sigmoid | Xavier / He: $\mathcal{N}(0, \sqrt{1/2})$ |

---

## 📂 Project Structure

```text
Life/
├── neural_network_practice_dataset.csv   # Breast cancer diagnostic dataset (30 features + label)
├── data.npz                              # Saved best model weights, biases & normalization params
├── pyproject.toml                        # Project dependencies and build configuration
├── uv.lock                               # Deterministic dependency lockfile
├── README.md                             # Documentation
└── src/
    └── life/
        ├── __init__.py                   # Core neural network, training loop, and CLI entry point
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python $\ge$ 3.10 (tested on Python 3.10 - 3.15)
- [`uv`](https://github.com/astral-sh/uv) (recommended) or standard `pip`

### 2. Installation

Clone the repository and set up the virtual environment:

```bash
# Clone the repository
git clone https://github.com/CodeStrikerMayank/LifeNN.git
cd LifeNN

# Using uv (fastest)
uv sync

# Or using standard venv + pip
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .
```

### 3. Training the Model

Run the training pipeline directly via the CLI entry point or python module:

```bash
# Using the installed CLI entry point
uv run life

# Or running the module directly
python -m life
```

The script will:
1. Load and split the dataset (70% Train, 15% Validation, 15% Test).
2. Standardize features based strictly on training distributions.
3. Train over 10,000 epochs using gradient descent.
4. Periodically report validation loss and accuracy.
5. Save the best-performing model checkpoint to `data.npz`.
6. Run a final unbiased evaluation against the unseen test set.

---

## 💻 Programmatic Usage & Inference

You can import and use the trained weights directly in your own scripts:

```python
import numpy as np
from life import predict, leaky_relu, sigmoid

# Load checkpoint
checkpoint = np.load("data.npz")
w1, b1 = checkpoint["w1"], checkpoint["b1"]
w2, b2 = checkpoint["w2"], checkpoint["b2"]
w3, b3 = checkpoint["w3"], checkpoint["b3"]
w4, b4 = checkpoint["w4"], checkpoint["b4"]
w5, b5 = checkpoint["w5"], checkpoint["b5"]
mean, std = checkpoint["mean"], checkpoint["std"]

# Example: normalize raw sample with training parameters
raw_sample = np.random.randn(1, 30)  # 30 patient diagnostic features
norm_sample = (raw_sample - mean) / (std + 1e-8)

# Run inference
probability = predict(norm_sample, w1, b1, w2, b2, w3, b3, w4, b4, w5, b5)
diagnosis = "Benign" if probability >= 0.5 else "Malignant"

print(f"Predicted Class: {diagnosis} (Confidence: {probability.item():.4f})")
```

---

## 📐 Mathematical Formulation

### Loss Function (Binary Cross-Entropy)
$$\mathcal{L}(y, \hat{y}) = -\frac{1}{m} \sum_{i=1}^m \left[ y^{(i)} \log(\hat{y}^{(i)} + \epsilon) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)} + \epsilon) \right]$$

### Output Layer Gradient
$$\frac{\partial \mathcal{L}}{\partial z_5} = a_5 - y$$

### Backpropagation Gradient Step (Layer $l$)
$$\delta_l = \left( \delta_{l+1} \cdot W_{l+1}^T \right) \odot \sigma'(z_l)$$
$$\frac{\partial \mathcal{L}}{\partial W_l} = \frac{1}{m} a_{l-1}^T \delta_l, \quad \frac{\partial \mathcal{L}}{\partial b_l} = \frac{1}{m} \sum_{i=1}^m \delta_l$$

---

## 📄 License

This project is licensed under the MIT License. Feel free to use, modify, and distribute for educational or personal projects.
