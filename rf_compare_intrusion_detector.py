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
    rf_compare_intrusion_detector.py

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
"""

import os
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


# -------------------------------------------------------------------
# Data loader
# -------------------------------------------------------------------

def load_sparta_cyber_data(path):
    import pandas as pd
    df = pd.read_csv(path)

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
            y_true_idx,
            y_pred_idx,
            display_labels=class_names,
            xticks_rotation=45,
            cmap="Blues",
            ax=ax,
        )
        ax.set_title(f"{model_name}")

    plt.tight_layout()
    fname = os.path.join(results_dir, "cm_all.png")
    plt.savefig(fname, dpi=300)
    plt.close()
    print(f"[INFO] Saved combined confusion matrices to {fname}")


def plot_roc_pr_curves(y_true_idx, prob_dict, class_names, results_dir):
    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true_idx, classes=range(n_classes))

    # Combined ROC (micro-average)
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
    fname = os.path.join(results_dir, "roc_comparison.png")
    plt.tight_layout()
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved ROC comparison to {fname}")

    # Combined PR (micro-average)
    plt.figure()
    for model_name, y_score in prob_dict.items():
        precision, recall, _ = precision_recall_curve(y_true_bin.ravel(), y_score.ravel())
        plt.plot(recall, precision, label=f"{model_name}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Comparison (micro-average)")
    plt.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))
    fname = os.path.join(results_dir, "pr_comparison.png")
    plt.tight_layout()
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved PR comparison to {fname}")


# -------------------------------------------------------------------
# Main pipeline
# -------------------------------------------------------------------

def run_rf_compare_pipeline(csv_path, results_dir="results"):

    # Create timestamped folder
    base_dir = ensure_results_dir(results_dir)
    run_ts = get_timestamp()
    results_dir = os.path.join(base_dir, run_ts)
    os.makedirs(results_dir, exist_ok=True)

    print(f"[INFO] Saving all outputs to: {results_dir}")

    # Load data
    X, y = load_sparta_cyber_data(csv_path)
    class_names = sorted(list(set(y)))
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    y_idx = np.array([class_to_idx[c] for c in y])

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_idx, test_size=0.3, random_state=42, stratify=y_idx
    )

    # SMOTE
    sm = SMOTE()
    X_train, y_train = sm.fit_resample(X_train, y_train)

    preds_idx_dict = {}
    prob_dict = {}

    y_true_idx = y_test
    y_true = np.array([class_names[i] for i in y_true_idx])

    # ---------------------------------------------------------------
    # MAIN MODEL: Random Forest
    # ---------------------------------------------------------------
    print("\n=== MAIN MODEL: RandomForest ===")
    rf = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42
    )
    rf.fit(X_train, y_train)
    rf_pred_idx = rf.predict(X_test)
    rf_pred = np.array([class_names[i] for i in rf_pred_idx])

    print("\n[RF Classification Report — Primary Model]")
    print(classification_report(y_true, rf_pred, digits=4))

    rf_report = classification_report(y_true, rf_pred, digits=4, output_dict=True)
    save_classification_report_csv(rf_report, os.path.join(results_dir, "rf_main_results.csv"))

    preds_idx_dict["RF (main)"] = rf_pred_idx
    prob_dict["RF (main)"] = rf.predict_proba(X_test)

    # ---------------------------------------------------------------
    # COMPARISON: XGBoost
    # ---------------------------------------------------------------
    if XGB_AVAILABLE:
        print("\n=== COMPARISON: XGBoost ===")
        xgb = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="mlogloss"
        )
        xgb.fit(X_train, y_train)
        xgb_pred_idx = xgb.predict(X_test)
        xgb_pred = np.array([class_names[i] for i in xgb_pred_idx])

        print("\n[XGBoost Classification Report]")
        print(classification_report(y_true, xgb_pred, digits=4))

        xgb_report = classification_report(y_true, xgb_pred, digits=4, output_dict=True)
        save_classification_report_csv(xgb_report, os.path.join(results_dir, "xgboost_results.csv"))

        preds_idx_dict["XGBoost"] = xgb_pred_idx
        prob_dict["XGBoost"] = xgb.predict_proba(X_test)
    else:
        print("[WARNING] XGBoost not installed. Skipping.")

    # ---------------------------------------------------------------
    # COMPARISON: Logistic Regression
    # ---------------------------------------------------------------
    print("\n=== COMPARISON: Logistic Regression ===")
    lr = LogisticRegression(max_iter=2000, class_weight="balanced")
    lr.fit(X_train, y_train)
    lr_pred_idx = lr.predict(X_test)
    lr_pred = np.array([class_names[i] for i in lr_pred_idx])

    print("\n[LogReg Classification Report]")
    print(classification_report(y_true, lr_pred, digits=4))

    lr_report = classification_report(y_true, lr_pred, digits=4, output_dict=True)
    save_classification_report_csv(lr_report, os.path.join(results_dir, "logreg_results.csv"))

    preds_idx_dict["LogReg"] = lr_pred_idx
    prob_dict["LogReg"] = lr.predict_proba(X_test)

    # ---------------------------------------------------------------
    # COMPARISON: SVM
    # ---------------------------------------------------------------
    print("\n=== COMPARISON: SVM (RBF) ===")
    svm = SVC(kernel="rbf", probability=True, class_weight="balanced")
    svm.fit(X_train, y_train)
    svm_pred_idx = svm.predict(X_test)
    svm_pred = np.array([class_names[i] for i in svm_pred_idx])

    print("\n[SVM Classification Report]")
    print(classification_report(y_true, svm_pred, digits=4))

    svm_report = classification_report(y_true, svm_pred, digits=4, output_dict=True)
    save_classification_report_csv(svm_report, os.path.join(results_dir, "svm_results.csv"))

    preds_idx_dict["SVM"] = svm_pred_idx
    prob_dict["SVM"] = svm.predict_proba(X_test)

    # ---------------------------------------------------------------
    # COMPARISON: MLP
    # ---------------------------------------------------------------
    print("\n=== COMPARISON: MLP ===")
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500)
    mlp.fit(X_train, y_train)
    mlp_pred_idx = mlp.predict(X_test)
    mlp_pred = np.array([class_names[i] for i in mlp_pred_idx])

    print("\n[MLP Classification Report]")
    print(classification_report(y_true, mlp_pred, digits=4))

    mlp_report = classification_report(y_true, mlp_pred, digits=4, output_dict=True)
    save_classification_report_csv(mlp_report, os.path.join(results_dir, "mlp_results.csv"))

    preds_idx_dict["MLP"] = mlp_pred_idx
    prob_dict["MLP"] = mlp.predict_proba(X_test)

    # ---------------------------------------------------------------
    # Plots
    # ---------------------------------------------------------------
    plot_confusion_matrices_combined(y_true_idx, preds_idx_dict, class_names, results_dir)
    plot_roc_pr_curves(y_true_idx, prob_dict, class_names, results_dir)


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
