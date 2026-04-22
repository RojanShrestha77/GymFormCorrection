from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_score, train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, accuracy_score, confusion_matrix
from sklearn.utils import resample
import pickle
import os

from config import N_FEATURES
import config

CSV_PATH = r"C:\Users\LOQ\Desktop\GymForm\lateral_raise_data.csv"
MODEL_PATH = r"C:\Users\LOQ\Desktop\GymForm\lateral_raise_model.pkl"

# load the csv
print("Loading data...")
df = pd.read_csv(CSV_PATH)
print(f"Total samples: {len(df)}")
print(f"  Correct:   {len(df[df['label'] == 'correct'])}")
print(f"  Incorrect: {len(df[df['label'] == 'incorrect'])}")

# -------- verify feature count matches config --------
actual_features = len(df.columns) - 1
if actual_features != N_FEATURES:
    raise ValueError(
        f"CSV has {actual_features} features but config expects {N_FEATURES}. "
        f"Re-run Script 1 to regenerate the CSV."
    )
print(f"Feature count verified: {actual_features} features (excluding label)")

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


print(f"Balanced dataset: {len(df_balanced)} samples")
print(f"  Correct:   {len(df_balanced[df_balanced['label'] == 'correct'])}")
print(f"  Incorrect: {len(df_balanced[df_balanced['label'] == 'incorrect'])}")

# split features and labels
x = df_balanced.drop("label", axis=1).values  # removes the label column
y = df_balanced["label"].values  # takes only the label column

# ----- train / test split ----
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
    # stratify makes sure both traingin and testing sets have similar proportions of categories.
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")

print("\nFeature columns:")
print(df.columns)

# ---- Train random forest ---
print("\nTraining model...")
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

print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
print("\nDetailed Report:")
print(classification_report(y_test, y_pred))


# --- cross validation -------
print("Running 5-Fold cross validation...")
skf = StratifiedKFold(n_splits=config.CV_FOLDS,
                      shuffle=True, random_state=config.RANDOM_STATE)
cv_scores = cross_val_score(model, x, y, cv=skf, scoring="accuracy", n_jobs=-1)
print(
    f"KFold Accuracy : {cv_scores.mean()*100:.2f}%  (+/- {cv_scores.std()*100:.2f}%)")
print(f"Per fold       : {[f'{s*100:.1f}%' for s in cv_scores]}")

# -------- confusion matrix ----------
cm = confusion_matrix(y_test, y_pred, labels=["correct", "incorrect"])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[
                              "correct", "incorrect"])
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, colorbar=False, camp="Blues")
ax.set_title(f"Confusion Matrix (accuracy {accuracy*100:.1f}% )")
plt.tight_layout()
cm_path = os.path.join(os.path.dirname(MODEL_PATH), "confusion_matrix.png")
plt.savefig(cm_path, dpi=150)
plt.close()
print(f"Confusion matrix saved to: {cm_path}")

# save the model
# wb means write binary, since we are writing a model file, model_path  is file namne model.pl
with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)


print(f"Model saved to: {MODEL_PATH}")
print("\nReady for Script 3 - Live Detection!")
