"""
Tourism Inference Script — matches assignment5training.py (v2)
- Loads preprocessor.joblib (ColumnTransformer pipeline) from artifacts/tourism_v2
- Loads threshold.json (F1-optimised) from artifacts/tourism_v2
- Loads model from examples/tourism_model.keras
  * Model was trained with Focal Loss + SMOTE — no action needed in inference,
    predict() does not use the loss function, only the learned weights
- Outputs metrics, confusion matrices, predictions CSV, and test_metrics.txt
"""

# =========================
# CONFIG
# =========================
TEST_PATH    = "data/tourism_testing.csv"
TARGET_COL   = "ProdTaken"

MODEL_PATH   = "examples/tourism_model.keras"
ARTIFACT_DIR = "artifacts/tourism_v2"
OUTPUT_DIR   = "data/output/assignment5"

# =========================
# IMPORTS
# =========================
import os
import json
import joblib
import numpy as np
import pandas as pd

from tensorflow import keras
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    roc_auc_score,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================
# Helper: detailed metrics
# =========================
def calculate_metrics(cm):
    tn, fp, fn, tp = cm.ravel()
    accuracy    = (tp + tn) / (tp + tn + fp + fn)
    precision   = tp / (tp + fp)  if (tp + fp)  > 0 else 0.0
    recall      = tp / (tp + fn)  if (tp + fn)  > 0 else 0.0
    f1          = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    specificity = tn / (tn + fp)  if (tn + fp)  > 0 else 0.0
    return {
        "accuracy": accuracy, "precision": precision,
        "recall": recall,     "f1_score": f1,
        "specificity": specificity,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


# =========================
# Helper: save confusion matrix
# =========================
def save_confusion_matrix(cm, class_names, save_path, normalized=False):
    if normalized:
        cm_plot = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        title, fmt = "Normalized Confusion Matrix", ".2f"
    else:
        cm_plot, title, fmt = cm, "Confusion Matrix", "d"

    plt.figure(figsize=(8, 6))
    if HAS_SEABORN:
        sns.heatmap(cm_plot, annot=True, fmt=fmt, cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names)
    else:
        plt.imshow(cm_plot, cmap="Blues")
        plt.colorbar()
        plt.xticks(range(len(class_names)), class_names)
        plt.yticks(range(len(class_names)), class_names)
        for i in range(cm_plot.shape[0]):
            for j in range(cm_plot.shape[1]):
                plt.text(j, i, format(cm_plot[i, j], fmt),
                         ha="center", va="center", color="black")

    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


# =========================
# MAIN
# =========================
def main():
    print("=" * 60)
    print("TOURISM INFERENCE  (v2 — ColumnTransformer pipeline)")
    print("=" * 60)

    # -------------------------
    # 1. Load artifacts
    # -------------------------
    preprocessor_path = os.path.join(ARTIFACT_DIR, "preprocessor.joblib")
    threshold_path    = os.path.join(ARTIFACT_DIR, "threshold.json")

    preprocessor = joblib.load(preprocessor_path)
    with open(threshold_path, "r") as f:
        threshold = json.load(f)["threshold"]

    print(f"\nLoaded preprocessor : {preprocessor_path}")
    print(f"Loaded threshold    : {threshold:.2f}")

    # -------------------------
    # 2. Load model
    # compile=False is intentional — the model was trained with a custom
    # focal loss, but inference only calls predict() which doesn't use
    # the loss function at all. We recompile with standard settings just
    # to keep evaluate() available if needed.
    # -------------------------
    print("\nLoading model...")
    model = keras.models.load_model(MODEL_PATH, compile=False)
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )
    print(f"Model loaded: {MODEL_PATH}")
    model.summary()

    # -------------------------
    # 3. Load test data
    # -------------------------
    print("\nLoading test data...")
    df_test = pd.read_csv(TEST_PATH)
    print(f"Test shape: {df_test.shape}")

    if TARGET_COL not in df_test.columns:
        raise ValueError(f"'{TARGET_COL}' not found in {TEST_PATH}")

    print("\nTarget distribution:")
    print(df_test[TARGET_COL].value_counts(dropna=False))

    # Clean target
    y_raw  = pd.to_numeric(df_test[TARGET_COL], errors="coerce")
    mask   = y_raw.notna()
    X_test = df_test.drop(columns=[TARGET_COL]).loc[mask].copy()
    y_test = y_raw.loc[mask].astype(int).values

    print(f"\nClean test samples: {len(X_test)}")

    # -------------------------
    # 4. Preprocess (transform only — never fit)
    # SMOTE was applied during training only, so inference just runs
    # the real test data through the saved ColumnTransformer pipeline.
    # -------------------------
    X_test_p = preprocessor.transform(X_test)
    print(f"Transformed feature count: {X_test_p.shape[1]}")

    # -------------------------
    # 5. Predict
    # -------------------------
    print("\nRunning inference...")
    y_prob = model.predict(X_test_p, verbose=0).flatten()
    y_pred = (y_prob >= threshold).astype(int)

    print(f"Predicted Taken     : {y_pred.sum()} / {len(y_pred)}")
    print(f"Predicted Not Taken : {(y_pred == 0).sum()} / {len(y_pred)}")

    # -------------------------
    # 6. Metrics
    # -------------------------
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy  : {accuracy:.4f} ({accuracy*100:.2f}%)")

    try:
        auc_score = roc_auc_score(y_test, y_prob)
        print(f"AUC Score : {auc_score:.4f}")
    except Exception:
        auc_score = None
        print("AUC Score : Could not calculate")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4, zero_division=0))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    m = calculate_metrics(cm)
    print("\nDetailed Metrics:")
    print(f"  Accuracy    : {m['accuracy']:.4f} ({m['accuracy']*100:.2f}%)")
    print(f"  Precision   : {m['precision']:.4f} ({m['precision']*100:.2f}%)")
    print(f"  Recall      : {m['recall']:.4f} ({m['recall']*100:.2f}%)")
    print(f"  F1-Score    : {m['f1_score']:.4f}")
    print(f"  Specificity : {m['specificity']:.4f} ({m['specificity']*100:.2f}%)")

    # -------------------------
    # 7. Confusion matrix plots
    # -------------------------
    save_confusion_matrix(cm, ["Not Taken", "Taken"],
                          os.path.join(OUTPUT_DIR, "confusion_matrix_test.png"))
    save_confusion_matrix(cm, ["Not Taken", "Taken"],
                          os.path.join(OUTPUT_DIR, "confusion_matrix_test_normalized.png"),
                          normalized=True)

    # -------------------------
    # 8. Predictions CSV
    # -------------------------
    results_df = df_test.loc[mask].copy()
    results_df["prediction_probability"] = np.round(y_prob, 4)
    results_df["predicted_label"]        = y_pred
    results_df["correct_prediction"]     = (
        results_df[TARGET_COL].astype(int) == results_df["predicted_label"]
    )

    pred_path = os.path.join(OUTPUT_DIR, "predictions.csv")
    results_df.to_csv(pred_path, index=False)
    print(f"\nPredictions saved : {pred_path}")

    # -------------------------
    # 9. Metrics txt
    # -------------------------
    metrics_path = os.path.join(OUTPUT_DIR, "test_metrics.txt")
    with open(metrics_path, "w") as f:
        f.write("TEST METRICS (Tourism v2)\n")
        f.write("=" * 60 + "\n")
        f.write(f"Model        : {MODEL_PATH}\n")
        f.write(f"Preprocessor : {preprocessor_path}\n")
        f.write(f"Test file    : {TEST_PATH}\n")
        f.write(f"Threshold    : {threshold:.2f}\n\n")
        f.write(f"Accuracy    : {m['accuracy']:.4f} ({m['accuracy']*100:.2f}%)\n")
        f.write(f"Precision   : {m['precision']:.4f} ({m['precision']*100:.2f}%)\n")
        f.write(f"Recall      : {m['recall']:.4f} ({m['recall']*100:.2f}%)\n")
        f.write(f"F1-Score    : {m['f1_score']:.4f}\n")
        f.write(f"Specificity : {m['specificity']:.4f} ({m['specificity']*100:.2f}%)\n")
        if auc_score is not None:
            f.write(f"AUC         : {auc_score:.4f}\n")
        f.write("\nConfusion Matrix:\n")
        f.write(str(cm) + "\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_test, y_pred, digits=4, zero_division=0))

    print(f"Metrics saved     : {metrics_path}")

    print("\n" + "=" * 60)
    print("Inference completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()