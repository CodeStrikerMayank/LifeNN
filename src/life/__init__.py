import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split as ts

FEATURE_NAMES = [
    'mean radius', 'mean texture', 'mean perimeter', 'mean area',
    'mean smoothness', 'mean compactness', 'mean concavity',
    'mean concave points', 'mean symmetry', 'mean fractal dimension',
    'radius error', 'texture error', 'perimeter error', 'area error',
    'smoothness error', 'compactness error', 'concavity error',
    'concave points error', 'symmetry error', 'fractal dimension error',
    'worst radius', 'worst texture', 'worst perimeter', 'worst area',
    'worst smoothness', 'worst compactness', 'worst concavity',
    'worst concave points', 'worst symmetry', 'worst fractal dimension'
]


def get_default_dataset_path() -> Path:
    """Finds the dataset path dynamically relative to the package or current directory."""
    # Check parent project directory first
    pkg_dir = Path(__file__).resolve().parent
    candidates = [
        pkg_dir.parent.parent / "neural_network_practice_dataset.csv",
        Path.cwd() / "neural_network_practice_dataset.csv",
        pkg_dir / "neural_network_practice_dataset.csv"
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def load_data(csv_path=None, test_size=0.30, val_size=0.50, random_state=42):
    """Loads and standardizes dataset, splitting into train, validation, and test sets."""
    if csv_path is None:
        csv_path = get_default_dataset_path()

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at {csv_path}")

    data = pd.read_csv(csv_path)
    X = data[FEATURE_NAMES].to_numpy()
    Y = data["diagnosis"].map({"malignant": 0, "benign": 1}).to_numpy().reshape(-1, 1)

    # Split into train and temporary holdout
    x_train, x_temp, y_train, y_temp = ts(
        X, Y, test_size=test_size, random_state=random_state, shuffle=True
    )
    # Split holdout into validation and test
    x_valid, x_test, y_valid, y_test = ts(
        x_temp, y_temp, test_size=val_size, random_state=random_state, shuffle=True
    )

    # Standardization: calculate statistics strictly from training set to avoid data leakage
    mean = np.mean(x_train, axis=0, keepdims=True)
    std = np.std(x_train, axis=0, keepdims=True)
    eps = 1e-8

    x_train_norm = (x_train - mean) / (std + eps)
    x_valid_norm = (x_valid - mean) / (std + eps)
    x_test_norm = (x_test - mean) / (std + eps)

    return (x_train_norm, y_train), (x_valid_norm, y_valid), (x_test_norm, y_test), (mean, std)


def sigmoid(z):
    """Numerically stable sigmoid activation."""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def leaky_relu(z, alpha=0.01):
    """Leaky ReLU activation function."""
    return np.maximum(alpha * z, z)


def leaky_relu_deriv(z, alpha=0.01):
    """Derivative of Leaky ReLU activation function."""
    return np.where(z > 0, 1.0, alpha)


# Aliases for backward compatibility
relu = leaky_relu
relu_deri = leaky_relu_deriv


def bce(y_true, y_pred):
    """Binary cross-entropy loss."""
    epsilon = 1e-15
    y_pred_clipped = np.clip(y_pred, epsilon, 1.0 - epsilon)
    return -np.mean(y_true * np.log(y_pred_clipped) + (1.0 - y_true) * np.log(1.0 - y_pred_clipped))


# Alias for backward compatibility
bse = bce


def compute_accuracy(y_true, y_pred, threshold=0.5):
    """Calculates classification accuracy in percentage."""
    predictions = (y_pred >= threshold).astype(float)
    return np.mean(predictions == y_true) * 100.0


def forward(X, w1, b1, w2, b2, w3, b3, w4, b4, w5, b5):
    """Performs forward propagation through the 5-layer network.
    
    Standard notation:
    z_l = linear pre-activation (W * a_(l-1) + b)
    a_l = activation output (f(z_l))
    """
    z1 = np.dot(X, w1) + b1
    a1 = leaky_relu(z1)

    z2 = np.dot(a1, w2) + b2
    a2 = leaky_relu(z2)

    z3 = np.dot(a2, w3) + b3
    a3 = leaky_relu(z3)

    z4 = np.dot(a3, w4) + b4
    a4 = leaky_relu(z4)

    z5 = np.dot(a4, w5) + b5
    a5 = sigmoid(z5)

    return z1, a1, z2, a2, z3, a3, z4, a4, z5, a5


def predict(X, w1, b1, w2, b2, w3, b3, w4, b4, w5, b5):
    """Computes output probabilities for input X."""
    *_, a5 = forward(X, w1, b1, w2, b2, w3, b3, w4, b4, w5, b5)
    return a5


def process(x_train, y_train, x_valid, y_valid, x_test=None, y_test=None,
            mean=None, std=None, learning_rate=0.05, epochs=10000,
            save_path="data.npz", seed=42):
    """Trains the 5-layer neural network with backpropagation and saves the best model checkpoint."""
    if seed is not None:
        np.random.seed(seed)

    # He / Xavier weight initialization
    w1 = np.random.randn(30, 16) * np.sqrt(2.0 / 30)
    b1 = np.zeros((1, 16))
    w2 = np.random.randn(16, 8) * np.sqrt(2.0 / 16)
    b2 = np.zeros((1, 8))
    w3 = np.random.randn(8, 4) * np.sqrt(2.0 / 8)
    b3 = np.zeros((1, 4))
    w4 = np.random.randn(4, 2) * np.sqrt(2.0 / 4)
    b4 = np.zeros((1, 2))
    w5 = np.random.randn(2, 1) * np.sqrt(1.0 / 2)
    b5 = np.zeros((1, 1))

    compare_loss = float("inf")
    best_epoch = 0
    best_val_accuracy = 0.0

    best_w1, best_b1 = w1.copy(), b1.copy()
    best_w2, best_b2 = w2.copy(), b2.copy()
    best_w3, best_b3 = w3.copy(), b3.copy()
    best_w4, best_b4 = w4.copy(), b4.copy()
    best_w5, best_b5 = w5.copy(), b5.copy()

    m = len(x_train)

    for epoch in range(epochs):
        # 1. Forward propagation
        z1, a1, z2, a2, z3, a3, z4, a4, z5, a5 = forward(
            x_train, w1, b1, w2, b2, w3, b3, w4, b4, w5, b5
        )

        # 2. Backward propagation
        # Layer 5 (Output)
        dz5 = a5 - y_train
        dw5 = (1.0 / m) * np.dot(a4.T, dz5)
        db5 = (1.0 / m) * np.sum(dz5, axis=0, keepdims=True)

        # Layer 4
        dz4 = np.dot(dz5, w5.T) * leaky_relu_deriv(z4)
        dw4 = (1.0 / m) * np.dot(a3.T, dz4)
        db4 = (1.0 / m) * np.sum(dz4, axis=0, keepdims=True)

        # Layer 3
        dz3 = np.dot(dz4, w4.T) * leaky_relu_deriv(z3)
        dw3 = (1.0 / m) * np.dot(a2.T, dz3)
        db3 = (1.0 / m) * np.sum(dz3, axis=0, keepdims=True)

        # Layer 2
        dz2 = np.dot(dz3, w3.T) * leaky_relu_deriv(z2)
        dw2 = (1.0 / m) * np.dot(a1.T, dz2)
        db2 = (1.0 / m) * np.sum(dz2, axis=0, keepdims=True)

        # Layer 1
        dz1 = np.dot(dz2, w2.T) * leaky_relu_deriv(z1)
        dw1 = (1.0 / m) * np.dot(x_train.T, dz1)
        db1 = (1.0 / m) * np.sum(dz1, axis=0, keepdims=True)

        # 3. Gradient descent updates
        w5 -= learning_rate * dw5
        b5 -= learning_rate * db5
        w4 -= learning_rate * dw4
        b4 -= learning_rate * db4
        w3 -= learning_rate * dw3
        b3 -= learning_rate * db3
        w2 -= learning_rate * dw2
        b2 -= learning_rate * db2
        w1 -= learning_rate * dw1
        b1 -= learning_rate * db1

        # Periodic validation & checkpointing
        if epoch % 1000 == 0 or epoch == epochs - 1:
            val_preds = predict(x_valid, w1, b1, w2, b2, w3, b3, w4, b4, w5, b5)
            val_loss = bce(y_valid, val_preds)
            val_acc = compute_accuracy(y_valid, val_preds)
            print(f"Epoch {epoch:5d}: Validation Loss = {val_loss:.4f} | Validation Accuracy = {val_acc:.2f}%")

            # Checkpoint the best model (using .copy() to prevent in-place mutation)
            if val_loss < compare_loss:
                compare_loss = val_loss
                best_epoch = epoch
                best_val_accuracy = val_acc
                best_w1, best_b1 = w1.copy(), b1.copy()
                best_w2, best_b2 = w2.copy(), b2.copy()
                best_w3, best_b3 = w3.copy(), b3.copy()
                best_w4, best_b4 = w4.copy(), b4.copy()
                best_w5, best_b5 = w5.copy(), b5.copy()

    print("\n--- Training Finished ---")
    print(f"Best Model from Epoch {best_epoch}: Validation Loss = {compare_loss:.4f} | Validation Accuracy = {best_val_accuracy:.2f}%")

    # Evaluate best model on test set if provided
    if x_test is not None and y_test is not None:
        test_preds = predict(x_test, best_w1, best_b1, best_w2, best_b2, best_w3, best_b3, best_w4, best_b4, best_w5, best_b5)
        test_loss = bce(y_test, test_preds)
        test_acc = compute_accuracy(y_test, test_preds)
        print(f"Test Set Evaluation: Test Loss = {test_loss:.4f} | Test Accuracy = {test_acc:.2f}%")

    # Save model weights, biases, best epoch, and normalization parameters
    save_data = {
        "w1": best_w1, "b1": best_b1,
        "w2": best_w2, "b2": best_b2,
        "w3": best_w3, "b3": best_b3,
        "w4": best_w4, "b4": best_b4,
        "w5": best_w5, "b5": best_b5,
        "epoch": best_epoch,
        "best_val_loss": compare_loss,
        "best_val_accuracy": best_val_accuracy
    }
    if mean is not None and std is not None:
        save_data["mean"] = mean
        save_data["std"] = std

    np.savez_compressed(save_path, **save_data)
    print(f"Model saved successfully to {save_path}")

    return {
        "weights": (best_w1, best_w2, best_w3, best_w4, best_w5),
        "biases": (best_b1, best_b2, best_b3, best_b4, best_b5),
        "best_epoch": best_epoch,
        "best_val_loss": compare_loss,
        "best_val_accuracy": best_val_accuracy
    }


def main():
    """Main CLI execution flow."""
    (x_train, y_train), (x_valid, y_valid), (x_test, y_test), (mean, std) = load_data()
    save_path = Path(__file__).resolve().parent.parent.parent / "data.npz"
    process(
        x_train, y_train,
        x_valid, y_valid,
        x_test, y_test,
        mean=mean, std=std,
        learning_rate=0.05,
        epochs=10000,
        save_path=save_path
    )


if __name__ == "__main__":
    main()
