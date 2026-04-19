import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils import resample
import pickle
import os

# -----------------------------------------------
CSV_PATH = r"C:\Users\LOQ\Desktop\GymForm\lateral_raise_data.csv"
MODEL_PATH = r"C:\Users\LOQ\Desktop\GymForm\lateral_raise_model.pkl"
# -----------------------------------------------

# --- Load the CSV ---
print("Loading data...")
df = pd.read_csv(CSV_PATH)
print(f"Total samples: {len(df)}")
print(f"  Correct:   {len(df[df['label'] == 'correct'])}")
print(f"  Incorrect: {len(df[df['label'] == 'incorrect'])}")

# --- Fix imbalance by upsampling incorrect samples ---
print("\nBalancing dataset...")
df_correct = df[df['label'] == 'correct']
df_incorrect = df[df['label'] == 'incorrect']

# Upsample incorrect to match correct count
# make more incorrect sample by duplicating them until they match the number of correct samples
df_incorrect_upsampled = resample(
    df_incorrect,
    replace=True,
    n_samples=len(df_correct),
    random_state=42
)

df_balanced = pd.concat([df_correct, df_incorrect_upsampled])
df_balanced = df_balanced.sample(
    frac=1, random_state=42).reset_index(drop=True)

print(f"Balanced dataset: {len(df_balanced)} samples")
print(f"  Correct:   {len(df_balanced[df_balanced['label'] == 'correct'])}")
print(f"  Incorrect: {len(df_balanced[df_balanced['label'] == 'incorrect'])}")

# --- Split features and labels ---
X = df_balanced.drop("label", axis=1).values
y = df_balanced["label"].values

# --- Train / Test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")

# --- Train Random Forest ---
print("\nTraining Random Forest model...")
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)
model.fit(X_train, y_train)

# --- Evaluate ---
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
print("\nDetailed Report:")
print(classification_report(y_test, y_pred))

# --- Save the model ---
with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

print(f"Model saved to: {MODEL_PATH}")
print("\nReady for Script 3 - Live Detection!")
