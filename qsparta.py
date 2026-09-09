# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 09:56:57 2026

@author: thanh

File: sparta_quantum_intrusion_detector.py

SPARTA-aligned hybrid quantum–classical intrusion detection prototype.

sparta_quantum_intrusion_detector.py
qsparta.py
sparta → aligns with the SPARTA space‑cyber threat framework

quantum → highlights the hybrid quantum–classical encoding

intrusion_detector → clearly states the function

.py → standard Python module
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
    Expected format (example):
    - Each row = one event or time window
    - Columns = engineered features from telemetry/logs
      (e.g., RF power stats, command frequency, error codes, link status)
    - Label column = SPARTA class (e.g., 'benign', 'uplink_jamming', 'spoofing')
    """
    import pandas as pd
    df = pd.read_csv(path)
    X = df.drop(columns=['label']).values
    y = df['label'].values
    return X, y

############################################
# 2. Quantum feature encoding (placeholder)
############################################

def quantum_feature_map(x, num_qubits):
    """
    Placeholder for quantum encoding.
    In practice, this would:
    - Normalize x
    - Map components to rotation angles or entangling patterns
    - Return a quantum state representation or expectation values
    Here we just return x for structural completeness.
    """
    # TODO: replace with real quantum encoding using your preferred framework
    return x

############################################
# 3. Hybrid quantum–classical model wrapper
############################################

class HybridQuantumClassifier:
    def __init__(self, num_qubits, num_classes, hidden_dim=32):
        self.num_qubits = num_qubits
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        # Placeholder classical weights
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
        # softmax
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
            # forward
            h = np.tanh(X_enc @ self.W1 + self.b1)
            logits = h @ self.W2 + self.b2
            # softmax
            exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

            # cross-entropy loss
            y_onehot = np.eye(self.num_classes)[y_idx]
            loss = -np.mean(np.sum(y_onehot * np.log(probs + 1e-9), axis=1))

            # gradients
            dlogits = (probs - y_onehot) / n
            dW2 = h.T @ dlogits
            db2 = np.sum(dlogits, axis=0)
            dh = dlogits @ self.W2.T
            dh_raw = dh * (1 - h**2)
            dW1 = X_enc.T @ dh_raw
            db1 = np.sum(dh_raw, axis=0)

            # update
            self.W2 -= lr * dW2
            self.b2 -= lr * db2
            self.W1 -= lr * dW1
            self.b1 -= lr * db1

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f}")

############################################
# 4. End-to-end SPARTA-aligned training demo
############################################

def run_sparta_quantum_pipeline(csv_path):
    X, y = load_sparta_cyber_data(csv_path)
    class_names = sorted(list(set(y)))
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    y_idx = np.array([class_to_idx[c] for c in y])

    # scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_idx, test_size=0.3, random_state=42, stratify=y_idx
    )

    num_qubits = X_train.shape[1]  # simple mapping: one feature per qubit
    num_classes = len(class_names)

    model = HybridQuantumClassifier(num_qubits=num_qubits, num_classes=num_classes)
    model.fit(X_train, y_train, epochs=50, lr=1e-2)

    y_pred = model.predict(X_test, class_names)
    y_true = np.array([class_names[i] for i in y_test])

    print(classification_report(y_true, y_pred, digits=4))

# Example usage (once you have a SPARTA-aligned CSV):
# run_sparta_quantum_pipeline("sparta_events.csv")
if __name__ == "__main__":
    # TODO: replace with real path
    csv_path = "sparta_events.csv"
    run_sparta_quantum_pipeline(csv_path)
