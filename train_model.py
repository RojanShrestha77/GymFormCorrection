"""
train_model.py = Train TCN model and xeport to TFLITE

INPUT:  lateral_raise_tcn_dataset.csv  (one row = one rep = 180 features)
OUTPUT: lateral_raise_model.tflite     (copy this to Flutter assets/)
        lateral_raise_model.keras      (backup, for retraining)
 
RUN ON COLAB IF SLOW:
  Upload your CSV, run this script, download the .tflite file.

 1. your csv file
 2. Load and reshape the data
 3.build the tcn model(the bran)
 4. train it 
 5. test how accurate it tis
 6. Shrink it to TFLite (phone-sized)
 later_raise_,odel.tflite -> copy to FLutter
"""

import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from datetime import datetime

from config import (
    CSV_PATH, TFLITE_MODEL_PATH, KERAS_MODEL_PATH,
    SEQ_LEN, N_FEATURES, N_CLASSES, LABELS,
    RANDOM_STATE, TEST_SIZE, EPOCHS, BATCH_SIZE,
    LEARNING_RATE, EARLY_STOPPING_PATIENCE,
    get_logger,
)

logger = get_logger("train_model")


# ========== LOAD DATA ===========
logger.info("=" * 60)
logger.info("GYMFORM TCN TRAINING")
logger.info("=" * 60)

logger.info(f"\nloading: {CSV_PATH}")
try:
    df = pd.read_csv(CSV_PATH)
except FileNotFoundError:
    logger.error(f"CSV not found: {CSV_PATH}")
    logger.error("Run collect_data.py first to ecord trainnig reps.")
    exit(1)

logger.info(f"Rows (reps): {len(df)}")
logger.info(
    f"Columns: {len(df.columns)} (1 label + {len(df.columns)-1} features)")
logger.info(f"\nLabel distribution:")

for lbl, cnt in df['label'].value_counts().items():
    logger.info(f" {lbl}: {cnt} reps")

# warn if dataset is too small
if len(df) < 80:
    logger.warning(
        f"Small dataset ({len(df)} reps). Aim for 140+ for reliable accuracy.")

# ========= PREPARE DATA =========
logger.info('\nPreparing data...')

# Feature matrix: reshape (n_reps, 180) -> (n_reps, 30, 6)
feature_cols = [c for c in df.columns if c != "label"]
X = df[feature_cols].values.astype(np.float32)
X = X.reshape(-1, SEQ_LEN, N_FEATURES)
logger.info(f"X shape: {X.shape}")


# Labels
label_to_idx = {lbl: i for i, lbl in enumerate(LABELS)}
y_idx = df['label'].map(label_to_idx).values  # integer class index

# One-hot for error head
y_form = (y_idx == 0).astype(np.float32)                  # 1=correct, 0=error
y_error = tf.keras.utils.to_categorical(y_idx, N_CLASSES)   # one-hot (n, 4)

logger.info(f"y_form shape:  {y_form.shape}  (binary: correct vs error)")
logger.info(f"y_error shape: {y_error.shape} (4-class: which error)")

# Train / test split
X_tr, X_te, yf_tr, yf_te, ye_tr, ye_te = train_test_split(
    X, y_form, y_error,
    test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_idx,
)
logger.info(f"\nTrain: {len(X_tr)} reps  |  Test: {len(X_te)} reps")

# Class weights (handles imbalanced data automatically)
cw = class_weight.compute_class_weight(
    "balanced", classes=np.unique(y_idx), y=y_idx
)
class_weights_dict = dict(enumerate(cw))
logger.info(f"Class weights: {class_weights_dict}")

# ============ build tcn model
logger.info("\nBuilding TCN model...")


def build_mobile_tcn(seq_len=SEQ_LEN, n_features=N_FEATURES, n_classes=N_CLASSES):
    inputs = tf.keras.Input(shape=(seq_len, n_features), name="pose_sequence")

    # TCN Block 1 — causal conv, dilation=1
    x = tf.keras.layers.Conv1D(
        32, kernel_size=3, padding="causal", dilation_rate=1,
        activation="relu", name="tcn_1")(inputs)
    x = tf.keras.layers.BatchNormalization(name="bn_1")(x)

    # TCN Block 2 — wider context, dilation=2
    x = tf.keras.layers.Conv1D(
        32, kernel_size=3, padding="causal", dilation_rate=2,
        activation="relu", name="tcn_2")(x)
    x = tf.keras.layers.BatchNormalization(name="bn_2")(x)
    x = tf.keras.layers.Dropout(0.2, name="drop_2")(x)

    # TCN Block 3 — even wider, dilation=4
    x = tf.keras.layers.Conv1D(
        32, kernel_size=3, padding="causal", dilation_rate=4,
        activation="relu", name="tcn_3")(x)
    x = tf.keras.layers.BatchNormalization(name="bn_3")(x)
    x = tf.keras.layers.Dropout(0.2, name="drop_3")(x)

    # Collapse time dimension
    x = tf.keras.layers.GlobalAveragePooling1D(name="gap")(x)
    x = tf.keras.layers.Dense(32, activation="relu", name="dense")(x)
    x = tf.keras.layers.Dropout(0.2, name="drop_out")(x)

    # Output 1: form quality — binary (correct vs error)
    form_out = tf.keras.layers.Dense(1, activation="sigmoid", name="form")(x)

    # Output 2: specific error class — 4-class softmax
    error_out = tf.keras.layers.Dense(n_classes, activation="softmax", name="error")(x)

    return tf.keras.Model(inputs, [form_out, error_out], name="GymFormTCN")


model = build_mobile_tcn()
model.summary(print_fn=logger.info)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss={
        "form": "binary_crossentropy",
        "error": "categorical_crossentropy",
    },
    loss_weights={"form": 1.0, "error": 0.8},
    metrics={
        "form": "accuracy",
        "error": "accuracy",
    },
)

# =========== train ================
logger.info(
    f"\nTraining for up to {EPOCHS} epochs (early stopping patience={EARLY_STOPPING_PATIENCE})...")

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1,

    ),  # stop training when the model stops improving on validation data
    # it watches val_loss(test/validation error)
    # if it doesnot improve for N epochs (patience)
    # it stops training
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",  # validation loss after every epoch
        factor=0.5,
        patience=5,
        min_lr=1e-5,
        verbose=1,
    ),  # if the model stops improving, reduce learning
]

start = datetime.now()
history = model.fit(
    X_tr,
    {"form": yf_tr, "error": ye_tr},
    validation_data=(X_te, {"form": yf_te, "error": ye_te}),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1,

)
duration = (datetime.now() - start).total_seconds()

epochs_run = len(history.history["loss"])
logger.info(f"\nTraining done: {epochs_run} epochs in {duration:.1f}s")

# ========== evaluate =============
logger.info("\nEvaluating on test set...")
results = model.evaluate(X_te, {"form": yf_te, "error": ye_te}, verbose=0)

# results order: total_loss, form_loss, error_loss, form_accuracy, error_accuracy
form_acc = results[3]
error_acc = results[4]

logger.info(f"Form accuracy: {form_acc*100:.2f}%")
logger.info(f"Error accuracy (which error):       {error_acc*100:.2f}%")


if form_acc < 0.80:
    logger.warning(
        "Form accuracy below 80%. Collect more data and reatain for better")
else:
    logger.info("Good form accuracy! Ready for TFLite conversion.")

# ======== save keras model =============
model.save(KERAS_MODEL_PATH)
logger.info(f"\nKeras model saved: {KERAS_MODEL_PATH}")


# ============== convert to TFLite ==============
logger.info("\nConverting to TFLite format...")

converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Float16 quantization: 4x smaller, ~2x faster, <1% accuracy loss
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

tflite_model = converter.convert()

with open(TFLITE_MODEL_PATH, "wb") as f:
    f.write(tflite_model)

size_kb = len(tflite_model) / 1024
logger.info(f"TFLite model saved: {TFLITE_MODEL_PATH}")
logger.info(f"Model size: {size_kb:.1f} KB  ({size_kb/1024:.2f} MB)")

if size_kb > 512:
    logger.warning(
        "Model size is above 512KB. Consider reducing model complexity for faster mobile inference.")

# ============= verify tflite model ================
logger.info("\nVerifying TFLite model...")


interp = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interp.allocate_tensors()

inp = interp.get_input_details()
outp = interp.get_output_details()


logger.info(f"Input shape:  {inp[0]['shape']}   dtype: {inp[0]['dtype']}")
logger.info(f"Output 0 (form):  {outp[0]['shape']}")
logger.info(f"Output 1 (error): {outp[1]['shape']}")

# run one sample
sample = X_te[0:1].astype(np.float32)
interp.set_tensor(inp[0]["index"], sample)
interp.invoke()

form_conf = interp.get_tensor(outp[0]["index"])[0][0]
error_pred = interp.get_tensor(outp[1]["index"])[0]
pred_label = LABELS[np.argmax(error_pred)]

logger.info(f"\nSample prediction:")
logger.info(
    f"  Form confidence: {form_conf:.3f}  ({'correct' if form_conf > 0.5 else 'error'})")
logger.info(
    f"  Error class:     {pred_label} ({max(error_pred)*100:.1f}% confident)")
logger.info(f"  True label:      {df['label'].iloc[0]}")


# =========== save training history ==============
report = {
    "timestamp":    datetime.now().isoformat(),
    "dataset_path": CSV_PATH,
    "total_reps":   len(df),
    "label_counts": df["label"].value_counts().to_dict(),
    "seq_len":      SEQ_LEN,
    "n_features":   N_FEATURES,
    "labels":       LABELS,
    "epochs_run":   epochs_run,
    "training_sec": round(duration, 1),
    "form_accuracy":  round(float(form_acc), 4),
    "error_accuracy": round(float(error_acc), 4),
    "tflite_path":  TFLITE_MODEL_PATH,
    "tflite_size_kb": round(size_kb, 1),
}

report_path = TFLITE_MODEL_PATH.replace(".tflite", "_report.json")
with open(report_path, "w") as f:
    json.dump(report, f, indent=2)
logger.info(f"\nReport saved: {report_path}")


# =============== summary ==================
logger.info("\n" + "=" * 60)
logger.info("TRAINING COMPLETE")
logger.info("=" * 60)
logger.info(f"Form accuracy:  {form_acc*100:.2f}%")
logger.info(f"Error accuracy: {error_acc*100:.2f}%")
logger.info(f"Model size:     {size_kb:.1f} KB")
logger.info(f"\nNEXT STEP:")
logger.info(f"Copy this file to your Flutter app:")
logger.info(f"  {TFLITE_MODEL_PATH}")
logger.info(f"  -> GymFormApplication/assets/models/lateral_raise_model.tflite")
logger.info("=" * 60)
