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
    rf_compare_intrusion_detector_timecompare.py

SPARTA / Aerospace Threat-Based intrusion detection pipeline (classical only).

Models:
    * RandomForest (MAIN MODEL)
    * XGBoost (comparison)
    * Logistic Regression (comparison)
    * SVM (RBF kernel) (comparison)
    * MLP (lightweight neural net) (comparison)

Features:
    - Auto-generate SPARTA synthetic dataset if missing
    - One-hot encoding of categorical fields
    - Standardization + SMOTE oversampling
    - Timestamped results folder per run
    - Confusion matrices (combined)
    - ROC curves (micro-average)
    - Precision–Recall curves (micro-average)
    - CSV results saved for each model
    - Training time (ms) + inference time (µs) for each model
    - Runtime vs Accuracy scatter plot
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd

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
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from imblearn.over_sampling import SMOTE

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

from generate_sparta_mock_data import generate_mock_sparta_events_csv


# -------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------

def ensure_results_dir(results_dir="results"):
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    return results_dir

def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def save_classification_report_csv(report_dict, out_path):
    df = pd.DataFrame(report_dict).transpose()
    df.to_csv(out_path, index=True)
    print(f"[INFO] Saved CSV results to {out_path}")

def measure_inference_time(model, X_test, n_runs=1000):
    sample = X_test[0].reshape(1, -1)
    model.predict(sample)  # warm-up

    start = time.perf_counter()
    for _ in range(n_runs):
        model.predict(sample)
    end = time.perf_counter()

    return (end - start) / n_runs * 1e6  # microseconds


def save_runtime_csv(model_name, train_ms, infer_us, results_dir):
    out_path = os.path.join(results_dir, f"{model_name}_runtime.csv")
    df = pd.DataFrame({
        "metric": ["training_time_ms", "inference_time_us"],
        "value": [train_ms, infer_us]
    })
    df.to_csv(out_path, index=False)
    print(f"[INFO] Saved runtime CSV to {out_path}")

    # Update combined summary
    summary_path = os.path.join(results_dir, "runtime_summary.csv")
    if not os.path.exists(summary_path):
        summary_df = pd.DataFrame(columns=["model", "training_time_ms", "inference_time_us"])
    else:
        summary_df = pd.read_csv(summary_path)

    summary_df.loc[len(summary_df)] = [model_name, train_ms, infer_us]
    summary_df.to_csv(summary_path, index=False)
    print(f"[INFO] Updated runtime summary at {summary_path}")


# -------------------------------------------------------------------
# Data loader (THIS WAS MISSING — NOW RESTORED)
# -------------------------------------------------------------------

def load_sparta_cyber_data(path):
    import pandas as pd
    df = pd.read_csv(path)

    # One-hot encode categorical fields if present
    cat_cols = [c for c in ["segment", "link_status"] if c in df.columns]
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols)

    X = df.drop(columns=["label"]).values
    y = df["label"].values
    return X, y


# -------------------------------------------------------------------
# Plotting
# -------------------------------------------------------------------

def plot_confusion_matrices_combined(y_true_idx, preds_idx_dict, class_names, results_dir):
    fig, axes = plt.subplots(1, len(preds_idx_dict), figsize=(5 * len(preds_idx_dict), 4))

    if len(preds_idx_dict) == 1:
        axes = [axes]

    for ax, (model_name, y_pred_idx) in zip(axes, preds_idx_dict.items()):
        ConfusionMatrixDisplay.from_predictions(
            y_true_idx, y_pred_idx,
            display_labels=class_names,
            xticks_rotation=45,
            cmap="Blues",
            ax=ax,
        )
        ax.set_title(model_name)

    plt.tight_layout()
    out = os.path.join(results_dir, "cm_all.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"[INFO] Saved combined confusion matrices to {out}")


def plot_roc_pr_curves(y_true_idx, prob_dict, class_names, results_dir):
    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true_idx, classes=range(n_classes))

    # ROC
    plt.figure()
    for model_name, y_score in prob_dict.items():
        fpr, tpr, _ = roc_curve(y_true_bin.ravel(), y_score.ravel())
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{model_name} (AUC={roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Comparison (micro-average)")
    plt.legend()
    out = os.path.join(results_dir, "roc_comparison.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved ROC comparison to {out}")

    # PR
    plt.figure()
    for model_name, y_score in prob_dict.items():
        precision, recall, _ = precision_recall_curve(y_true_bin.ravel(), y_score.ravel())
        plt.plot(recall, precision, label=model_name)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Comparison (micro-average)")
    plt.legend()
    out = os.path.join(results_dir, "pr_comparison.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved PR comparison to {out}")


def plot_runtime_vs_accuracy(runtime_dict, accuracy_dict, results_dir):
    plt.figure(figsize=(7, 5))

    for model_name in runtime_dict:
        x = runtime_dict[model_name]
        y = accuracy_dict[model_name] * 100
        plt.scatter(x, y, s=120)
        plt.text(x + 5, y, model_name, fontsize=10)

    plt.xlabel("Inference Time (µs per sample)")
    plt.ylabel("Accuracy (%)")
    plt.title("Runtime vs Accuracy Comparison")
    plt.grid(True, linestyle="--", alpha=0.5)

    out = os.path.join(results_dir, "runtime_vs_accuracy.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"[INFO] Saved runtime vs accuracy plot to {out}")


# -------------------------------------------------------------------
# Main pipeline
# -------------------------------------------------------------------

def run_rf_compare_pipeline(csv_path, results_dir="results"):

    base_dir = ensure_results_dir(results_dir)
    run_ts = get_timestamp()
    results_dir = os.path.join(base_dir, run_ts)
    os.makedirs(results_dir, exist_ok=True)

    print(f"[INFO] Saving all outputs to: {results_dir}")

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

    preds_idx_dict = {}
    prob_dict = {}
    runtime_dict = {}
    accuracy_dict = {}

    y_true_idx = y_test
    y_true = np.array([class_names[i] for i in y_true_idx])

    # ---------------------------------------------------------------
    # Random Forest
    # ---------------------------------------------------------------
    print("\n=== MAIN MODEL: RandomForest ===")
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42)

    t0 = time.perf_counter()
    rf.fit(X_train, y_train)
    t1 = time.perf_counter()
    rf_train_ms = (t1 - t0) * 1000
    print(f"[RF] Training time: {rf_train_ms:.3f} ms")

    rf_pred_idx = rf.predict(X_test)
    rf_pred = np.array([class_names[i] for i in rf_pred_idx])

    rf_inf_us = measure_inference_time(rf, X_test)
    print(f"[RF] Avg inference time: {rf_inf_us:.3f} µs")

    rf_report = classification_report(y_true, rf_pred, digits=4, output_dict=True)
    save_classification_report_csv(rf_report, os.path.join(results_dir, "rf_main_results.csv"))
    save_runtime_csv("rf_main", rf_train_ms, rf_inf_us, results_dir)

    preds_idx_dict["RF (main)"] = rf_pred_idx
    prob_dict["RF (main)"] = rf.predict_proba(X_test)
    runtime_dict["RF (main)"] = rf_inf_us
    accuracy_dict["RF (main)"] = rf_report["accuracy"]

    # ---------------------------------------------------------------
    # Logistic Regression
    # ---------------------------------------------------------------
    print("\n=== COMPARISON: Logistic Regression ===")
    lr = LogisticRegression(max_iter=2000, class_weight="balanced")

    t0 = time.perf_counter()
    lr.fit(X_train, y_train)
    t1 = time.perf_counter()
    lr_train_ms = (t1 - t0) * 1000
    print(f"[LogReg] Training time: {lr_train_ms:.3f} ms")

    lr_pred_idx = lr.predict(X_test)
    lr_pred = np.array([class_names[i] for i in lr_pred_idx])

    lr_inf_us = measure_inference_time(lr, X_test)
    print(f"[LogReg] Avg inference time: {lr_inf_us:.3f} µs")

    lr_report = classification_report(y_true, lr_pred, digits=4, output_dict=True)
    save_classification_report_csv(lr_report, os.path.join(results_dir, "logreg_results.csv"))
    save_runtime_csv("logreg", lr_train_ms, lr_inf_us, results_dir)

    preds_idx_dict["LogReg"] = lr_pred_idx
    prob_dict["LogReg"] = lr.predict_proba(X_test)
    runtime_dict["LogReg"] = lr_inf_us
    accuracy_dict["LogReg"] = lr_report["accuracy"]

    # ---------------------------------------------------------------
    # SVM
    # ---------------------------------------------------------------
    print("\n=== COMPARISON: SVM (RBF) ===")
    svm = SVC(kernel="rbf", probability=True, class_weight="balanced")

    t0 = time.perf_counter()
    svm.fit(X_train, y_train)
    t1 = time.perf_counter()
    svm_train_ms = (t1 - t0) * 1000
    print(f"[SVM] Training time: {svm_train_ms:.3f} ms")

    svm_pred_idx = svm.predict(X_test)
    svm_pred = np.array([class_names[i] for i in svm_pred_idx])

    svm_inf_us = measure_inference_time(svm, X_test)
    print(f"[SVM] Avg inference time: {svm_inf_us:.3f} µs")

    svm_report = classification_report(y_true, svm_pred, digits=4, output_dict=True)
    save_classification_report_csv(svm_report, os.path.join(results_dir, "svm_results.csv"))
    save_runtime_csv("svm", svm_train_ms, svm_inf_us, results_dir)

    preds_idx_dict["SVM"] = svm_pred_idx
    prob_dict["SVM"] = svm.predict_proba(X_test)
    runtime_dict["SVM"] = svm_inf_us
    accuracy_dict["SVM"] = svm_report["accuracy"]

    # ---------------------------------------------------------------
    # MLP
    # ---------------------------------------------------------------
    print("\n=== COMPARISON: MLP ===")
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500)

    t0 = time.perf_counter()
    mlp.fit(X_train, y_train)
    t1 = time.perf_counter()
    mlp_train_ms = (t1 - t0) * 1000
    print(f"[MLP] Training time: {mlp_train_ms:.3f} ms")

    mlp_pred_idx = mlp.predict(X_test)
    mlp_pred = np.array([class_names[i] for i in mlp_pred_idx])

    mlp_inf_us = measure_inference_time(mlp, X_test)
    print(f"[MLP] Avg inference time: {mlp_inf_us:.3f} µs")

    mlp_report = classification_report(y_true, mlp_pred, digits=4, output_dict=True)
    save_classification_report_csv(mlp_report, os.path.join(results_dir, "mlp_results.csv"))
    save_runtime_csv("mlp", mlp_train_ms, mlp_inf_us, results_dir)

    preds_idx_dict["MLP"] = mlp_pred_idx
    prob_dict["MLP"] = mlp.predict_proba(X_test)
    runtime_dict["MLP"] = mlp_inf_us
    accuracy_dict["MLP"] = mlp_report["accuracy"]

    # ---------------------------------------------------------------
    # Plots
    # ---------------------------------------------------------------
    plot_confusion_matrices_combined(y_true_idx, preds_idx_dict, class_names, results_dir)
    plot_roc_pr_curves(y_true_idx, prob_dict, class_names, results_dir)
    plot_runtime_vs_accuracy(runtime_dict, accuracy_dict, results_dir)


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

if __name__ == "__main__":
    csv_path = "sparta_events.csv"

    if not os.path.exists(csv_path):
        print("[INFO] Dataset not found. Generating synthetic SPARTA dataset...")
        generate_mock_sparta_events_csv(csv_path)
        print("[INFO] Dataset generated.")

    run_rf_compare_pipeline(csv_path, results_dir="results")
