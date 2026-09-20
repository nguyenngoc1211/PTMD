"""
train_xgboost.py — Train XGBoost classifier, evaluate on test set.
Outputs:
  results/models/xgboost_model.joblib
  results/tables/model_metrics.csv
  results/figures/02_confusion_matrix.png
"""
from common import CONFIG, setup
import os, argparse, json, warnings
import pandas as pd, numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
RANDOM_STATE = CONFIG["seed"]

def main():
    setup()
    for d in ["results/models", "results/tables", "results/figures"]:
        os.makedirs(d, exist_ok=True)

    split = joblib.load("results/intermediate/train_test_split.joblib")
    X_train, X_test = split["X_train"], split["X_test"]
    y_train, y_test = split["y_train"], split["y_test"]
    feature_names = split["feature_names"]

    print(f"[train] Train={X_train.shape[0]}, Test={X_test.shape[0]}, Features={X_train.shape[1]}")

    # ── XGBoost training ──────────────────────────────────────
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=CONFIG["threads"],
    )
    model.fit(X_train, y_train)
    print("[train] XGBoost trained.")

    # ── Predictions ───────────────────────────────────────────
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # ── Metrics ───────────────────────────────────────────────
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    metrics = {
        "accuracy": acc, "precision": prec, "recall": rec,
        "f1": f1, "roc_auc": auc,
        "n_train": X_train.shape[0], "n_test": X_test.shape[0],
        "n_features": X_train.shape[1],
    }
    pd.DataFrame([metrics]).to_csv("results/tables/model_metrics.csv", index=False)
    print(f"[train] Metrics: acc={acc:.6f}, f1={f1:.6f}, auc={auc:.6f}")

    report = classification_report(y_test, y_pred, target_names=["Benign", "Malware"])
    print(report)
    open("results/tables/classification_report.txt", "w").write(report)

    # ── Save model + data ─────────────────────────────────────
    joblib.dump({
        "model": model,
        "features": feature_names,
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "y_pred": y_pred, "y_prob": y_prob,
        "feature_groups": split["feature_groups"],
    }, "results/models/xgboost_model.joblib")
    print("[train] Saved model + data to results/models/xgboost_model.joblib")

    # ── Figure 02: Confusion matrix ───────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Benign", "Malware"],
                yticklabels=["Benign", "Malware"], ax=ax)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=13)
    plt.tight_layout()
    plt.savefig("results/figures/02_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[train] Saved 02_confusion_matrix.png")

if __name__ == "__main__":
    main()
