# -*- coding: utf-8 -*-
"""
###################################################
###################################################
The active research areas at TAN's Laboratory: 
                 Time-sensitive networking (T),
                   AI-driven cybersecurity (A),
            NextG communication networking (N),
Time-series Analysis via Network science (TAN)
###################################################
###################################################
@author:
    Van Le - Tan Le
 sparta_quantum_intrusion_detector_updated.py.
SPARTA / Aerospace Threat-Based hybrid quantum–classical intrusion detection prototype (updated).

Features:
- Auto-generate SPARTA synthetic dataset if missing (via generate_mock_sparta_events_csv)
- One-hot encoding of categorical fields
- Standardization + SMOTE oversampling
- Models:
    * RandomForest (classical baseline)
    * HybridQuantumClassifier (NumPy, quantum-inspired)
    * QuantumEnhancedClassifier (PennyLane VQC)
- Class-weighted training for HQC and QE
- Confusion matrices (combined comparison)
- ROC curves per class and per model
- Precision–Recall curves per class and per model
- Combined ROC and PR comparison panels (RF vs HQC vs QE) with legend outside
- Auto-created results/ folder with compact timestamped filenames
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (
    classification_report,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    precision_recall_curve,
)
from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import SMOTE

# Import dataset generator
from generate_sparta_mock_data import generate_mock_sparta_events_csv

# -------------------------------------------------------------------
# 1. Utilities: results dir + timestamp
# -------------------------------------------------------------------

def ensure_results_dir(results_dir="results"):
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    return results_dir

def get_timestamp():
    # Compact style: YYYYMMDD_HHMMSS
    return datetime.now().strftime("%Y%m%d_%H%M%S")

# -------------------------------------------------------------------
# 2. Data interface (SPARTA-aligned events)
# -------------------------------------------------------------------

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

    cat_cols = [c for c in ["segment", "link_status"] if c in df.columns]
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols)

    X = df.drop(columns=["label"]).values
    y = df["label"].values
    return X, y

# -------------------------------------------------------------------
# 3. Quantum feature encoding (placeholder)
# -------------------------------------------------------------------

def quantum_feature_map(x, num_qubits):
    x = x / (np.linalg.norm(x) + 1e-9)
    return x

# -------------------------------------------------------------------
# 4. Hybrid Quantum-Inspired Classifier (NumPy)
# -------------------------------------------------------------------

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
                print(f"[Baseline HQC] Epoch {epoch+1}/{epochs} - Loss: {loss:.4f}")

# -------------------------------------------------------------------
# 5. Quantum-Enhanced Classifier (PennyLane)
# -------------------------------------------------------------------

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
            loss_accum = 0.0
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
                y_onehot[y] = 1.0
                w = class_weights[y]

                loss = w * (-np.sum(y_onehot * np.log(probs + 1e-9)))
                loss_accum += loss

                dlogits = w * (probs - y_onehot)
                grads_Wc += np.outer(q_out, dlogits)
                grads_bc += dlogits

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

            print(f"[Quantum QE] Epoch {epoch+1}/{epochs} - Loss: {loss_accum/n:.4f}")

# -------------------------------------------------------------------
# 6. Plotting helpers
# -------------------------------------------------------------------

def plot_confusion_matrices_combined(y_true_idx, preds_idx_dict, class_names, results_dir):
    ts = get_timestamp()
    fig, axes = plt.subplots(1, len(preds_idx_dict), figsize=(5 * len(preds_idx_dict), 4))

    if len(preds_idx_dict) == 1:
        axes = [axes]

    for ax, (model_name, y_pred_idx) in zip(axes, preds_idx_dict.items()):
        ConfusionMatrixDisplay.from_predictions(
            y_true_idx,
            y_pred_idx,
            display_labels=class_names,
            xticks_rotation=45,
            cmap="Blues",
            ax=ax,
        )
        ax.set_title(f"{model_name} Confusion Matrix")

    plt.tight_layout()
    fname = os.path.join(results_dir, f"cm_all_{ts}.png")
    plt.savefig(fname, dpi=300)
    print(f"[INFO] Saved combined confusion matrices to {fname}")
    plt.close()

def plot_roc_pr_curves(y_true_idx, prob_dict, class_names, results_dir):
    ts = get_timestamp()
    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true_idx, classes=range(n_classes))

    # Individual ROC and PR per model and per class
    for model_name, y_score in prob_dict.items():
        for c in range(n_classes):
            # ROC
            fpr, tpr, _ = roc_curve(y_true_bin[:, c], y_score[:, c])
            roc_auc = auc(fpr, tpr)
            plt.figure()
            plt.plot(fpr, tpr, label=f"Class {class_names[c]} (AUC={roc_auc:.3f})")
            plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"ROC - {model_name} - {class_names[c]}")
            plt.legend(loc="lower right")
            fname = os.path.join(results_dir, f"roc_{model_name}_class{c}_{ts}.png")
            plt.tight_layout()
            plt.savefig(fname, dpi=300)
            plt.close()

            # PR
            precision, recall, _ = precision_recall_curve(y_true_bin[:, c], y_score[:, c])
            plt.figure()
            plt.plot(recall, precision, label=f"Class {class_names[c]}")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title(f"Precision-Recall - {model_name} - {class_names[c]}")
            plt.legend(loc="lower left")
            fname = os.path.join(results_dir, f"pr_{model_name}_class{c}_{ts}.png")
            plt.tight_layout()
            plt.savefig(fname, dpi=300)
            plt.close()

    # Combined ROC (micro-average) overlay
    plt.figure()
    for model_name, y_score in prob_dict.items():
        fpr, tpr, _ = roc_curve(y_true_bin.ravel(), y_score.ravel())
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{model_name} (AUC={roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Comparison (micro-average)")
    plt.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))
    fname = os.path.join(results_dir, f"roc_comparison_{ts}.png")
    plt.tight_layout()
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    print(f"[INFO] Saved ROC comparison to {fname}")
    plt.close()

    # Combined PR (micro-average) overlay
    plt.figure()
    for model_name, y_score in prob_dict.items():
        precision, recall, _ = precision_recall_curve(y_true_bin.ravel(), y_score.ravel())
        plt.plot(recall, precision, label=f"{model_name}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Comparison (micro-average)")
    plt.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))
    fname = os.path.join(results_dir, f"pr_comparison_{ts}.png")
    plt.tight_layout()
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    print(f"[INFO] Saved PR comparison to {fname}")
    plt.close()

# -------------------------------------------------------------------
# 7. End-to-end pipeline
# -------------------------------------------------------------------

def run_sparta_quantum_pipeline(csv_path, use_quantum=True, results_dir="results"):
    results_dir = ensure_results_dir(results_dir)

    X, y = load_sparta_cyber_data(csv_path)
    class_names = sorted(list(set(y)))
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    y_idx = np.array([class_to_idx[c] for c in y])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_idx, test_size=0.3, random_state=42, stratify=y_idx
    )

    sm = SMOTE()
    X_train, y_train = sm.fit_resample(X_train, y_train)

    num_features = X_train.shape[1]
    num_classes = len(class_names)

    y_true_idx = y_test
    y_true = np.array([class_names[i] for i in y_true_idx])

    preds_idx_dict = {}
    prob_dict = {}

    # RandomForest baseline
    print("\n=== RandomForest Baseline ===")
    rf = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42
    )
    rf.fit(X_train, y_train)
    rf_pred_idx = rf.predict(X_test)
    rf_pred = np.array([class_names[i] for i in rf_pred_idx])
    print(classification_report(y_true, rf_pred, digits=4))
    preds_idx_dict["RandomForest"] = rf_pred_idx
    prob_dict["RandomForest"] = rf.predict_proba(X_test)

    # Hybrid Quantum-Inspired
    print("\n=== HybridQuantumClassifier (NumPy) ===")
    hq = HybridQuantumClassifier(num_qubits=num_features, num_classes=num_classes)
    hq.fit(X_train, y_train, epochs=50, lr=1e-2)
    hq_probs = hq.predict_proba(X_test)
    hq_pred_idx = np.argmax(hq_probs, axis=1)
    hq_pred = np.array([class_names[i] for i in hq_pred_idx])
    print(classification_report(y_true, hq_pred, digits=4))
    preds_idx_dict["HybridQuantum"] = hq_pred_idx
    prob_dict["HybridQuantum"] = hq_probs

    # Quantum-Enhanced
    if use_quantum and PENNYLANE_AVAILABLE:
        print("\n=== QuantumEnhancedClassifier (PennyLane) ===")
        qe = QuantumEnhancedClassifier(
            num_features=num_features,
            num_classes=num_classes,
            num_qubits=min(4, num_features),
            n_layers=2,
        )
        qe.fit(X_train, y_train, epochs=5, lr=5e-2)
        qe_probs = qe.predict_proba(X_test)
        qe_pred_idx = np.argmax(qe_probs, axis=1)
        qe_pred = np.array([class_names[i] for i in qe_pred_idx])
        print(classification_report(y_true, qe_pred, digits=4))
        preds_idx_dict["QuantumEnhanced"] = qe_pred_idx
        prob_dict["QuantumEnhanced"] = qe_probs
    elif use_quantum:
        print("[WARNING] PennyLane not installed. Skipping QuantumEnhancedClassifier.")

    # Plots
    if preds_idx_dict:
        plot_confusion_matrices_combined(y_true_idx, preds_idx_dict, class_names, results_dir)
    if prob_dict:
        plot_roc_pr_curves(y_true_idx, prob_dict, class_names, results_dir)

# -------------------------------------------------------------------
# 8. Main
# -------------------------------------------------------------------

if __name__ == "__main__":
    csv_path = "sparta_events.csv"

    if not os.path.exists(csv_path):
        print("[INFO] Dataset not found. Generating synthetic SPARTA dataset...")
        generate_mock_sparta_events_csv(csv_path)
        print("[INFO] Dataset generated.")

    run_sparta_quantum_pipeline(csv_path, use_quantum=True, results_dir="results")
