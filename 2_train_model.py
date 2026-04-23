from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_score, train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, accuracy_score, confusion_matrix
from sklearn.utils import resample
import pickle
import os
from config import CSV_PATH, MODEL_PATH, N_FEATURES, get_logger
import config

# Set up logging
logger = get_logger("train_model")

# load the csv with error handling
logger.info("Loading training data...")
try:
    df = pd.read_csv(CSV_PATH)
    logger.info(f"Total samples: {len(df)}")
    logger.info(f"  Correct:   {len(df[df['label'] == 'correct'])}")
    logger.info(f"  Incorrect: {len(df[df['label'] == 'incorrect'])}")
except FileNotFoundError:
    logger.error(f"CSV file not found: {CSV_PATH}")
    logger.error("Run Script 1 (1_extract_landmarks.py) first to generate the dataset.")
    exit(1)
except pd.errors.EmptyDataError:
    logger.error(f"CSV file is empty: {CSV_PATH}")
    exit(1)
except Exception as e:
    logger.error(f"Failed to load CSV: {e}")
    exit(1)

# -------- verify feature count matches config --------
actual_features = len(df.columns) - 1
if actual_features != N_FEATURES:
    error_msg = (
        f"CSV has {actual_features} features but config expects {N_FEATURES}. "
        f"Re-run Script 1 to regenerate the CSV."
    )
    logger.error(error_msg)
    raise ValueError(error_msg)
logger.info(f"Feature count verified: {actual_features} features (excluding label)")

# fixing the imbalance by upsamopling the incrrect sampled
df_correct = df[df['label'] == 'correct']
df_incorrect = df[df['label'] == 'incorrect']


# upsample incorrect to match the current count
df_incorrect_upsampled = resample(
    df_incorrect,
    replace=True,
    n_samples=len(df_correct),
    random_state=config.RANDOM_STATE
)

df_balanced = pd.concat([df_correct, df_incorrect_upsampled])
df_balanced = df_balanced.sample(frac=1, random_state=config.RANDOM_STATE).reset_index(
    drop=True)  # this shuffles the data
# sample(frac=1) means:take 100% data but in random order


logger.info(f"Balanced dataset: {len(df_balanced)} samples")
logger.info(f"  Correct:   {len(df_balanced[df_balanced['label'] == 'correct'])}")
logger.info(f"  Incorrect: {len(df_balanced[df_balanced['label'] == 'incorrect'])}")

# split features and labels
x = df_balanced.drop("label", axis=1).values  # removes the label column
y = df_balanced["label"].values  # takes only the label column

# ----- train / test split ----
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
    # stratify makes sure both traingin and testing sets have similar proportions of categories.
)

logger.info(f"\nTraining samples: {len(X_train)}")
logger.info(f"Testing samples:  {len(X_test)}")

logger.info("\nFeature columns:")
logger.info(str(df.columns.tolist()))

# ---- Train random forest ---
logger.info("\nTraining model...")
model = RandomForestClassifier(
    n_estimators=config.N_ESTIMATORS,  # creatinig 100 decision trees
    random_state=config.RANDOM_STATE,
    class_weight="balanced",  # balances the inbalcned data
    n_jobs=-1  # uses all CPU cores to speed up training
)
model.fit(X_train, y_train)


# evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

logger.info(f"\nModel Accuracy: {accuracy * 100:.2f}%")
logger.info("\nDetailed Report:")
logger.info("\n" + classification_report(y_test, y_pred))


# --- cross validation -------
logger.info("Running 5-Fold cross validation...")
skf = StratifiedKFold(n_splits=config.CV_FOLDS,
                      shuffle=True, random_state=config.RANDOM_STATE)
cv_scores = cross_val_score(model, x, y, cv=skf, scoring="accuracy", n_jobs=-1)
logger.info(
    f"KFold Accuracy : {cv_scores.mean()*100:.2f}%  (+/- {cv_scores.std()*100:.2f}%)")
logger.info(f"Per fold       : {[f'{s*100:.1f}%' for s in cv_scores]}")

# -------- confusion matrix ----------
cm = confusion_matrix(y_test, y_pred, labels=["correct", "incorrect"])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[
                              "correct", "incorrect"])
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title(f"Confusion Matrix (accuracy {accuracy*100:.1f}% )")
plt.tight_layout()
cm_path = os.path.join(os.path.dirname(MODEL_PATH), "confusion_matrix.png")
plt.savefig(cm_path, dpi=150)
plt.close()
logger.info(f"Confusion matrix saved to: {cm_path}")

# save the model with error handling
logger.info("Saving model...")
try:
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Model saved to: {MODEL_PATH}")
    logger.info("\nReady for Script 3 - Live Detection!")
except PermissionError:
    logger.error(f"Permission denied: Cannot write to {MODEL_PATH}")
    exit(1)
except Exception as e:
    logger.error(f"Failed to save model: {e}")
    exit(1)
