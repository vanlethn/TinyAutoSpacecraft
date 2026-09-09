# -*- coding: utf-8 -*-
"""
SPARTA / Aerospace Threat-Based hybrid quantum–classical intrusion detection prototype.
Automatically:
- Generates SPARTA synthetic dataset if missing
- Loads + preprocesses data (one-hot, scaling)
- Applies SMOTE oversampling
- Runs 3 models:
    * RandomForest baseline
    * HybridQuantumClassifier (NumPy)
    * QuantumEnhancedClassifier (PennyLane)
- Plots confusion matrices
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import SMOTE

# Import dataset generator
from generate_sparta_mock_data import generate_mock_sparta_events_csv

############################################
# 1. Data interface (SPARTA-aligned events)
############################################

def load_sparta_cyber_data(path):
    import pandas as pd
    df = pd.read_csv(path)

    # One-hot encode categorical fields
    cat_cols = [c for c in ["segment", "link_status"] if c in df.columns]
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols)

    X = df.drop(columns=["label"]).values
    y = df["label"].values
    return X, y

############################################
# 2. Quantum feature encoding (placeholder)
############################################

def quantum_feature_map(x, num_qubits):
    x = x / (np.linalg.norm(x) + 1e-9)
    return x

############################################
# 3A. Hybrid Quantum-Inspired Classifier (NumPy)
############################################

class HybridQuantumClassifier:
    def __init__(self, num_qubits, num_classes, hidden_dim=32):
        self.num_qubits = num_qubits
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.W1 = np.random.randn(num_qubits, hidden_dim) * 0.1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, num_classes) * 0.1
        self.b2 = np.zeros(num_classes)

    def _forward(self, x_enc):
        h = np.tanh(x_enc @ self.W1 + self.b1)
        logits = h @ self.W2 + self.b2
        return logits

    def predict_proba(self, X):
        X_enc = np.array([quantum_feature_map(x, self.num_qubits) for x in X])
        logits = self._forward(X_enc)
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    def predict(self, X, class_names):
        probs = self.predict_proba(X)
        idx = np.argmax(probs, axis=1)
        return np.array([class_names[i] for i in idx])

    def fit(self, X, y_idx, epochs=50, lr=1e-2):
        X_enc = np.array([quantum_feature_map(x, self.num_qubits) for x in X])
        n = X_enc.shape[0]

        # Class weights
        class_counts = np.bincount(y_idx)
        class_weights = class_counts.max() / class_counts

        for epoch in range(epochs):
            h = np.tanh(X_enc @ self.W1 + self.b1)
            logits = h @ self.W2 + self.b2
            exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

            y_onehot = np.eye(self.num_classes)[y_idx]
            sample_weights = class_weights[y_idx]
            loss = -np.mean(sample_weights * np.sum(y_onehot * np.log(probs + 1e-9), axis=1))

            dlogits = (probs - y_onehot) * sample_weights[:, None] / n
            dW2 = h.T @ dlogits
            db2 = np.sum(dlogits, axis=0)
            dh = dlogits @ self.W2.T
            dh_raw = dh * (1 - h**2)
            dW1 = X_enc.T @ dh_raw
            db1 = np.sum(dh_raw, axis=0)

            self.W2 -= lr * dW2
            self.b2 -= lr * db2
            self.W1 -= lr * dW1
            self.b1 -= lr * db1

            if (epoch + 1) % 10 == 0:
                print(f"[Baseline] Epoch {epoch+1}/{epochs} - Loss: {loss:.4f}")

############################################
# 3B. Quantum-Enhanced Classifier (PennyLane)
############################################

try:
    import pennylane as qml
    PENNYLANE_AVAILABLE = True
except ImportError:
    PENNYLANE_AVAILABLE = False

class QuantumEnhancedClassifier:
    def __init__(self, num_features, num_classes, num_qubits=4, n_layers=2):
        if not PENNYLANE_AVAILABLE:
            raise ImportError("PennyLane not installed.")

        self.num_features = num_features
        self.num_classes = num_classes
        self.num_qubits = min(num_qubits, num_features)
        self.n_layers = n_layers

        self.dev = qml.device("default.qubit", wires=self.num_qubits)

        @qml.qnode(self.dev)
        def circuit(x, weights):
            for i in range(self.num_qubits):
                qml.RY(x[i], wires=i)
            for l in range(self.n_layers):
                for i in range(self.num_qubits):
                    qml.RZ(weights[l, i, 0], wires=i)
                    qml.RY(weights[l, i, 1], wires=i)
                for i in range(self.num_qubits - 1):
                    qml.CNOT(wires=[i, i+1])
            return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]

        self.circuit = circuit
        self.weights = 0.01 * np.random.randn(self.n_layers, self.num_qubits, 2)
        self.W_classical = np.random.randn(self.num_qubits, self.num_classes) * 0.1
        self.b_classical = np.zeros(self.num_classes)

    def _forward_single(self, x):
        x = x / (np.linalg.norm(x) + 1e-9)
        x = x[:self.num_qubits]
        q_out = np.array(self.circuit(x, self.weights))
        return q_out @ self.W_classical + self.b_classical

    def predict_proba(self, X):
        logits = np.vstack([self._forward_single(x) for x in X])
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    def predict(self, X, class_names):
        probs = self.predict_proba(X)
        idx = np.argmax(probs, axis=1)
        return np.array([class_names[i] for i in idx])

    def fit(self, X, y_idx, epochs=5, lr=5e-2):
        n = X.shape[0]
        class_counts = np.bincount(y_idx)
        class_weights = class_counts.max() / class_counts

        for epoch in range(epochs):
            loss_accum = 0
            grads_w = np.zeros_like(self.weights)
            grads_Wc = np.zeros_like(self.W_classical)
            grads_bc = np.zeros_like(self.b_classical)

            for i in range(n):
                x = X[i]
                y = y_idx[i]
                x_norm = x / (np.linalg.norm(x) + 1e-9)
                x_enc = x_norm[:self.num_qubits]

                q_out = np.array(self.circuit(x_enc, self.weights))
                logits = q_out @ self.W_classical + self.b_classical
                exp_logits = np.exp(logits - np.max(logits))
                probs = exp_logits / np.sum(exp_logits)

                y_onehot = np.zeros(self.num_classes)
                y_onehot[y] = 1
                w = class_weights[y]

                loss = w * (-np.sum(y_onehot * np.log(probs + 1e-9)))
                loss_accum += loss

                dlogits = w * (probs - y_onehot)
                grads_Wc += np.outer(q_out, dlogits)
                grads_bc += dlogits

                # Parameter-shift gradients
                for l in range(self.n_layers):
                    for q in range(self.num_qubits):
                        for p in range(2):
                            shift = np.zeros_like(self.weights)
                            shift[l, q, p] = np.pi / 2
                            plus = np.array(self.circuit(x_enc, self.weights + shift))
                            minus = np.array(self.circuit(x_enc, self.weights - shift))
                            dq = 0.5 * (plus - minus)
                            grads_w[l, q, p] += dq @ (self.W_classical @ dlogits)

            self.weights -= lr * grads_w / n
            self.W_classical -= lr * grads_Wc / n
            self.b_classical -= lr * grads_bc / n

            print(f"[Quantum] Epoch {epoch+1}/{epochs} - Loss: {loss_accum/n:.4f}")

############################################
# 4. Confusion Matrix Plot
############################################

def plot_confusion_matrix(y_true, y_pred, class_names, title):
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=class_names, xticks_rotation=45
    )
    plt.title(title)
    plt.tight_layout()
    plt.show()

############################################
# 5. End-to-end Pipeline
############################################

def run_sparta_quantum_pipeline(csv_path, use_quantum=False):
    X, y = load_sparta_cyber_data(csv_path)
    class_names = sorted(list(set(y)))
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    y_idx = np.array([class_to_idx[c] for c in y])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_idx, test_size=0.3, random_state=42, stratify=y_idx
    )

    # SMOTE oversampling
    sm = SMOTE()
    X_train, y_train = sm.fit_resample(X_train, y_train)

    num_features = X_train.shape[1]
    num_classes = len(class_names)

    # === RandomForest Baseline ===
    print("\n=== RandomForest Baseline ===")
    rf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
    rf.fit(X_train, y_train)
    rf_pred_idx = rf.predict(X_test)
    rf_pred = np.array([class_names[i] for i in rf_pred_idx])
    y_true = np.array([class_names[i] for i in y_test])
    print(classification_report(y_true, rf_pred, digits=4))
    plot_confusion_matrix(y_true, rf_pred, class_names, "RandomForest Confusion Matrix")

    # === Hybrid Quantum-Inspired Model ===
    print("\n=== HybridQuantumClassifier (NumPy) ===")
    hq = HybridQuantumClassifier(num_qubits=num_features, num_classes=num_classes)
    hq.fit(X_train, y_train, epochs=50, lr=1e-2)
    hq_pred = hq.predict(X_test, class_names)
    print(classification_report(y_true, hq_pred, digits=4))
    plot_confusion_matrix(y_true, hq_pred, class_names, "HybridQuantumClassifier Confusion Matrix")

    # === Quantum-Enhanced Model ===
    if use_quantum and PENNYLANE_AVAILABLE:
        print("\n=== QuantumEnhancedClassifier (PennyLane) ===")
        qe = QuantumEnhancedClassifier(
            num_features=num_features,
            num_classes=num_classes,
            num_qubits=min(4, num_features),
            n_layers=2,
        )
        qe.fit(X_train, y_train, epochs=5, lr=5e-2)
        qe_pred = qe.predict(X_test, class_names)
        print(classification_report(y_true, qe_pred, digits=4))
        plot_confusion_matrix(y_true, qe_pred, class_names, "QuantumEnhancedClassifier Confusion Matrix")
    elif use_quantum:
        print("[WARNING] PennyLane not installed. Skipping quantum model.")

############################################
# 6. Main
############################################

if __name__ == "__main__":
    csv_path = "sparta_events.csv"

    # Auto-generate dataset if missing
    if not os.path.exists(csv_path):
        print("[INFO] Dataset not found. Generating synthetic SPARTA dataset...")
        generate_mock_sparta_events_csv(csv_path)
        print("[INFO] Dataset generated.")

    print("\n=== Running Baseline Model ===")
    run_sparta_quantum_pipeline(csv_path, use_quantum=False)

    print("\n=== Running Quantum-Enhanced Model ===")
    run_sparta_quantum_pipeline(csv_path, use_quantum=True)
