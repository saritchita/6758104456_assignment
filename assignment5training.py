# src/assignment5training.py
# Train + Validate using: data/tourism_data.csv
# Keep tourism_testing.csv hidden for inference
#
# Improvements applied:
# - ColumnTransformer pipeline (impute + encode + scale) saved as single artifact
# - Focal Loss  (replaces binary_crossentropy + class_weight)
# - SMOTE oversampling on train set only
# - F1 Callback — EarlyStopping monitors val F1 directly
# - Fixed architecture bug (layers now chain properly)

# =========================
# CONFIG
# =========================
TRAIN_PATH   = "data/tourism_data.csv"
TARGET_COL   = "ProdTaken"
RANDOM_STATE = 42

MODEL_PATH   = "examples/tourism_model.keras"
ARTIFACT_DIR = "artifacts/tourism_v2"
OUTPUT_DIR   = "data/output/assignment5"

VAL_SIZE     = 0.15
EPOCHS       = 300
BATCH_SIZE   = 64
LEARNING_RATE = 1e-3

PATIENCE_ES  = 35
PATIENCE_LR  = 12
MIN_LR       = 5e-4  

THRESH_MIN   = 0.10
THRESH_MAX   = 0.80
THRESH_STEP  = 0.01

# =========================
# IMPORTS
# =========================
import os
import json
import joblib
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from imblearn.over_sampling import SMOTE

os.makedirs(ARTIFACT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


# =========================
# FOCAL LOSS
# Why: binary_crossentropy treats every sample equally.
# With ~18% positive class, the model gets lazy predicting "Not Taken"
# and still achieves low loss. Focal loss multiplies the loss of
# easy/correct predictions by (1-p)^gamma, shrinking their contribution
# so the model is forced to focus on hard, misclassified examples.
# alpha=0.75 adds extra weight to the minority class on top of that.
# =========================
def focal_loss(gamma=2.0, alpha=0.75):
    def loss(y_true, y_pred):
        y_true  = tf.cast(y_true, tf.float32)
        y_pred  = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        bce     = -y_true * tf.math.log(y_pred) \
                  - (1 - y_true) * tf.math.log(1 - y_pred)
        p_t     = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_t = y_true * alpha  + (1 - y_true) * (1 - alpha)
        return tf.reduce_mean(alpha_t * tf.pow(1 - p_t, gamma) * bce)
    return loss


# =========================
# LOAD & CLEAN
# =========================
print("Loading training data...")
df = pd.read_csv(TRAIN_PATH)
print("Raw shape:", df.shape)
assert TARGET_COL in df.columns, f"Target '{TARGET_COL}' not found."

y_num = pd.to_numeric(df[TARGET_COL], errors="coerce")
mask  = y_num.notna()
df    = df.loc[mask].copy()
df[TARGET_COL] = y_num.loc[mask].astype(int)

print(f"Dropped {int((~mask).sum())} rows with missing/invalid target.")
print("Clean shape:", df.shape)
print("Target distribution:\n", df[TARGET_COL].value_counts())

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL].values


# =========================
# TRAIN / VAL SPLIT
# Split BEFORE preprocessing so val never influences fit stats
# =========================
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=VAL_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)
print(f"\nTrain: {X_train.shape}  Val: {X_val.shape}")


# =========================
# PREPROCESSOR — fit on TRAIN only
# ColumnTransformer handles imputation + encoding + scaling in one object.
# Saving this single object to disk means inference can never
# accidentally re-fit or mis-align columns.
# =========================
num_cols = X_train.select_dtypes(include="number").columns.tolist()
cat_cols = X_train.select_dtypes(include="object").columns.tolist()

print(f"\nNumeric features   : {len(num_cols)}")
print(f"Categorical features: {len(cat_cols)}")

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
])

categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot",  OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipe, num_cols),
    ("cat", categorical_pipe, cat_cols),
], remainder="drop")

print("\nFitting preprocessor on TRAIN only...")
X_train_p = preprocessor.fit_transform(X_train)
X_val_p   = preprocessor.transform(X_val)

joblib.dump(preprocessor, os.path.join(ARTIFACT_DIR, "preprocessor.joblib"))
print(f"Preprocessor saved. Transformed feature count: {X_train_p.shape[1]}")


# =========================
# SMOTE — applied to TRAIN only, AFTER preprocessing
# Why: SMOTE generates synthetic minority-class samples by interpolating
# between real ones in feature space. Applying it after scaling ensures
# the synthetic points are created in the same normalised space the
# model will see. Val set stays real and untouched so metrics are honest.
# NOTE: because SMOTE balances the classes, we no longer need class_weight.
# =========================
#print(f"\nBefore SMOTE — {dict(zip(*np.unique(y_train, return_counts=True)))}")
#sm = SMOTE(random_state=RANDOM_STATE)
#X_train_p, y_train = sm.fit_resample(X_train_p, y_train)
#print(f"After  SMOTE — {dict(zip(*np.unique(y_train, return_counts=True)))}")


# =========================
# MODEL — fixed sequential architecture
# Previous bug: Dense(256) and Dense(128) both connected to `inputs`
# instead of the previous layer, creating accidental parallel branches.
# Now each layer feeds into the next as intended.
# =========================
def build_model(n_features: int) -> keras.Model:
    l2 = keras.regularizers.l2(3e-4)

    inputs = keras.Input(shape=(n_features,))

    x = keras.layers.Dense(128, kernel_regularizer=l2)(inputs)  
    x = keras.layers.BatchNormalization()(x)   
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.Dropout(0.3)(x)

    x = keras.layers.Dense(128, kernel_regularizer=l2)(x)    
    x = keras.layers.BatchNormalization()(x)   
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.Dropout(0.15)(x)

    x = keras.layers.Dense(64, kernel_regularizer=l2)(x)
    x = keras.layers.BatchNormalization()(x)   
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.Dropout(0.2)(x)

    outputs = keras.layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inputs, outputs)

model = build_model(X_train_p.shape[1])

# Focal loss replaces binary_crossentropy — no class_weight needed
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss=focal_loss(gamma=2.0, alpha=0.75),
    metrics=[
        "accuracy",
        keras.metrics.AUC(name="auc"),
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
    ],
)

print("\nModel summary:")
model.summary()


# =========================
# F1 CALLBACK
# Why: val_auc is a decent proxy but F1 is what we actually care about.
# This callback sweeps thresholds each epoch and logs the best achievable
# val F1 so EarlyStopping and ReduceLROnPlateau respond to F1 directly.
# =========================
class F1Callback(keras.callbacks.Callback):
    def __init__(self, val_data):
        super().__init__()
        self.X_val, self.y_val = val_data

    def on_epoch_end(self, epoch, logs=None):
        proba   = self.model.predict(self.X_val, verbose=0).flatten()
        best_f1 = max(
            f1_score(self.y_val, (proba >= t).astype(int), zero_division=0)
            for t in np.arange(THRESH_MIN, THRESH_MAX, 0.05)
        )
        logs["val_f1"] = best_f1


# =========================
# CALLBACKS
# =========================
callbacks = [
    F1Callback(val_data=(X_val_p, y_val)),
    keras.callbacks.EarlyStopping(
        monitor="val_f1",
        patience=PATIENCE_ES,
        restore_best_weights=True,
        mode="max",
        verbose=1,
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_f1",
        factor=0.5,
        patience=PATIENCE_LR,
        min_lr=MIN_LR,
        mode="max",
        verbose=1,
    ),
    keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(ARTIFACT_DIR, "best_model.keras"),
        monitor="val_f1",
        save_best_only=True,
        mode="max",
        verbose=1,
    ),
]


# =========================
# TRAIN
# =========================
print("\nTraining...")
history = model.fit(
    X_train_p, y_train,
    validation_data=(X_val_p, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1,
)


# =========================
# SAVE TRAINING HISTORY (for plotting in inference)
# =========================
history_path = os.path.join(ARTIFACT_DIR, "history.json")
with open(history_path, "w") as f:
    json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f, indent=2)
print(f"Training history saved: {history_path}")


# =========================
# THRESHOLD OPTIMISATION ON VAL (fine-grained sweep)
# =========================
val_proba = model.predict(X_val_p, verbose=0).flatten()

best_t, best_f1 = 0.5, -1.0
for t in np.arange(THRESH_MIN, THRESH_MAX, THRESH_STEP):
    preds = (val_proba >= t).astype(int)
    f     = f1_score(y_val, preds, zero_division=0)
    if f > best_f1:
        best_f1, best_t = f, t

print(f"\nOptimal threshold (VAL): {best_t:.2f}  →  Val F1: {best_f1:.4f}")


# =========================
# FINAL VAL REPORT
# =========================
val_pred = (val_proba >= best_t).astype(int)
cm = confusion_matrix(y_val, val_pred)

val_loss, val_acc, val_auc, val_prec, val_rec = model.evaluate(
    X_val_p, y_val, verbose=0
)
print(f"\nVal Loss : {val_loss:.4f}")
print(f"Val Acc  : {val_acc:.4f}")
print(f"Val AUC  : {val_auc:.4f}")
print(f"Val Prec : {val_prec:.4f}")
print(f"Val Rec  : {val_rec:.4f}")
print(f"\nClassification Report (VAL, threshold={best_t:.2f}):")
print(classification_report(y_val, val_pred, digits=4, zero_division=0))
print("Confusion Matrix (VAL):")
print(cm)


# =========================
# SAVE ARTIFACTS
# =========================
model.save(MODEL_PATH)

with open(os.path.join(ARTIFACT_DIR, "threshold.json"), "w") as f:
    json.dump({"threshold": float(best_t)}, f, indent=2)

report_path = os.path.join(OUTPUT_DIR, "val_report.txt")
with open(report_path, "w") as f:
    f.write("VALIDATION REPORT\n")
    f.write("=" * 60 + "\n")
    f.write(f"Val Loss  : {val_loss:.4f}\n")
    f.write(f"Val Acc   : {val_acc:.4f}\n")
    f.write(f"Val AUC   : {val_auc:.4f}\n")
    f.write(f"Threshold : {best_t:.2f}\n")
    f.write(f"Val F1    : {best_f1:.4f}\n\n")
    f.write("Confusion Matrix:\n")
    f.write(str(cm) + "\n\n")
    f.write("Classification Report:\n")
    f.write(classification_report(y_val, val_pred, digits=4, zero_division=0))

print("\n--- Saved artifacts ---")
print(f"  Model        : {MODEL_PATH}")
print(f"  Best model   : {ARTIFACT_DIR}/best_model.keras")
print(f"  Preprocessor : {ARTIFACT_DIR}/preprocessor.joblib")
print(f"  Threshold    : {ARTIFACT_DIR}/threshold.json")
print(f"  History      : {ARTIFACT_DIR}/history.json")
print(f"  Val report   : {report_path}")