# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 09:56:57 2026

@author: thanh

File: sparta_quantum_intrusion_detector.py

SPARTA / Aerospace Threat-Based hybrid quantum–classical intrusion detection prototype.
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

############################################
# 1. Data interface (SPARTA-aligned events)
############################################

def load_sparta_cyber_data(path):
    """
    Expected format:
    - Each row = one event or time window
    - Columns = engineered features from telemetry/logs
      (e.g., RF power stats, command behavior, attitude/thermal deviations, link status)
    - Categorical columns: 'segment', 'link_status'
    - Label column: 'label' (e.g., 'benign', 'uplink_jamming', 'spoofing', ...)
    """
    import pandas as pd
    df = pd.read_csv(path)

    # One-hot encode categorical fields
    if "segment" in df.columns or "link_status" in df.columns:
        df = pd.get_dummies(df, columns=[c for c in ["segment", "link_status"] if c in df.columns])

    X = df.drop(columns=['label']).values
    y = df['label'].values
    return X, y

############################################
# 2. Quantum feature encoding (classical placeholder)
############################################

def quantum_feature_map(x, num_qubits):
    """
    Physics-informed placeholder quantum encoding.
    Normalizes the feature vector to mimic amplitude encoding.
    """
    x = x / (np.linalg.norm(x) + 1e-9)
    return x

############################################
# 3A. Baseline hybrid quantum–classical (NumPy only)
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
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        return probs

    def predict(self, X, class_names):
        probs = self.predict_proba(X)
        idx = np.argmax(probs, axis=1)
        return np.array([class_names[i] for i in idx])

    def fit(self, X, y_idx, epochs=50, lr=1e-2):
        X_enc = np.array([quantum_feature_map(x, self.num_qubits) for x in X])
        n = X_enc.shape[0]
        for epoch in range(epochs):
            h = np.tanh(X_enc @ self.W1 + self.b1)
            logits = h @ self.W2 + self.b2
            exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

            y_onehot = np.eye(self.num_classes)[y_idx]
            loss = -np.mean(np.sum(y_onehot * np.log(probs + 1e-9), axis=1))

            dlogits = (probs - y_onehot) / n
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
# 3B. Quantum-enhanced classifier (PennyLane)
############################################

try:
    import pennylane as qml
    PENNYLANE_AVAILABLE = True
except ImportError:
    PENNYLANE_AVAILABLE = False

class QuantumEnhancedClassifier:
    """
    Simple variational quantum classifier using PennyLane.
    - Uses angle encoding of features into rotations
    - Outputs expectation values mapped to class logits
    """

    def __init__(self, num_features, num_classes, num_qubits=4, n_layers=2):
        if not PENNYLANE_AVAILABLE:
            raise ImportError("PennyLane is not installed in this environment.")

        self.num_features = num_features
        self.num_classes = num_classes
        self.num_qubits = min(num_qubits, num_features)
        self.n_layers = n_layers

        self.dev = qml.device("default.qubit", wires=self.num_qubits)

        @qml.qnode(self.dev)
        def circuit(x, weights):
            # Angle encoding of first num_qubits features
            for i in range(self.num_qubits):
                qml.RY(x[i], wires=i)

            # Variational layers
            for l in range(self.n_layers):
                for i in range(self.num_qubits):
                    qml.RZ(weights[l, i, 0], wires=i)
                    qml.RY(weights[l, i, 1], wires=i)
                for i in range(self.num_qubits - 1):
                    qml.CNOT(wires=[i, i+1])

            # Measure expectation values
            return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]

        self.circuit = circuit
        self.weights = 0.01 * np.random.randn(self.n_layers, self.num_qubits, 2)
        self.W_classical = np.random.randn(self.num_qubits, self.num_classes) * 0.1
        self.b_classical = np.zeros(self.num_classes)

    def _forward_single(self, x):
        x = x / (np.linalg.norm(x) + 1e-9)
        x = x[:self.num_qubits]
        q_out = np.array(self.circuit(x, self.weights))
        logits = q_out @ self.W_classical + self.b_classical
        return logits

    def predict_proba(self, X):
        logits_list = []
        for x in X:
            logits = self._forward_single(x)
            logits_list.append(logits)
        logits = np.vstack(logits_list)
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        return probs

    def predict(self, X, class_names):
        probs = self.predict_proba(X)
        idx = np.argmax(probs, axis=1)
        return np.array([class_names[i] for i in idx])

    def fit(self, X, y_idx, epochs=20, lr=1e-2):
        n = X.shape[0]
        for epoch in range(epochs):
            grads_w = np.zeros_like(self.weights)
            grads_Wc = np.zeros_like(self.W_classical)
            grads_bc = np.zeros_like(self.b_classical)
            loss_accum = 0.0

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
                y_onehot[y] = 1.0
                loss = -np.sum(y_onehot * np.log(probs + 1e-9))
                loss_accum += loss

                dlogits = probs - y_onehot
                grads_Wc += np.outer(q_out, dlogits)
                grads_bc += dlogits

                # Parameter-shift gradients for quantum weights
                for l in range(self.n_layers):
                    for q in range(self.num_qubits):
                        for p in range(2):
                            shift = np.zeros_like(self.weights)
                            shift[l, q, p] = np.pi / 2
                            plus = np.array(self.circuit(x_enc, self.weights + shift))
                            minus = np.array(self.circuit(x_enc, self.weights - shift))
                            dq = 0.5 * (plus - minus)
                            grads_w[l, q, p] += dq @ (self.W_classical @ dlogits)

            loss_accum /= n
            grads_w /= n
            grads_Wc /= n
            grads_bc /= n

            self.weights -= lr * grads_w
            self.W_classical -= lr * grads_Wc
            self.b_classical -= lr * grads_bc

            print(f"[Quantum] Epoch {epoch+1}/{epochs} - Loss: {loss_accum:.4f}")

############################################
# 4. End-to-end SPARTA-aligned training demo
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

    num_features = X_train.shape[1]
    num_classes = len(class_names)

    if use_quantum and PENNYLANE_AVAILABLE:
        print("Using quantum-enhanced classifier (PennyLane).")
        model = QuantumEnhancedClassifier(
            num_features=num_features,
            num_classes=num_classes,
            num_qubits=min(4, num_features),
            n_layers=2,
        )
        model.fit(X_train, y_train, epochs=5, lr=5e-2)
    else:
        if use_quantum and not PENNYLANE_AVAILABLE:
            print("PennyLane not available, falling back to baseline classifier.")
        print("Using baseline hybrid classifier (NumPy).")
        model = HybridQuantumClassifier(num_qubits=num_features, num_classes=num_classes)
        model.fit(X_train, y_train, epochs=50, lr=1e-2)

    y_pred = model.predict(X_test, class_names)
    y_true = np.array([class_names[i] for i in y_test])

    print(classification_report(y_true, y_pred, digits=4))

############################################
# 5. Main
############################################

if __name__ == "__main__":
    csv_path = "sparta_events.csv"
    # Baseline:
    run_sparta_quantum_pipeline(csv_path, use_quantum=False)
    # Quantum-enhanced (requires PennyLane):
    run_sparta_quantum_pipeline(csv_path, use_quantum=True)
