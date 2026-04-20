import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils import resample
import pickle
import os

CSV_PATH = r"C:\Users\LOQ\Desktop\GymForm\lateral_raise_data.csv"
MODEL_PATH = r"C:\Users\LOQ\Desktop\GymForm\lateral_raise_model.pkl"

# load the csv
print("Loading data...")
df = pd.read_csv(CSV_PATH)
print(f"Total samples: {len(df)}")
print(f"  Correct:   {len(df[df['label'] == 'correct'])}")
print(f"  Incorrect: {len(df[df['label'] == 'incorrect'])}")

# fixing the imbalance by upsamopling the incrrect sampled
df_correct = df[df['label'] == 'correct']
df_incorrect = df[df['label'] == 'incorrect']


# upsample incorrect to match the current count
df_incorrect_upsampled = resample(
    df_incorrect,
    replace=True,
    n_samples=len(df_correct),
    random_state=42
)

df_balanced = pd.concat([df_correct, df_incorrect_upsampled])
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(
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

# ---- Train random forest ---
print("\nTraining model...")
model = RandomForestClassifier(
    n_estimators=100,  # creatinig 100 decision trees
    random_state=42,
    class_weight="balanced"  # balances the inbalcned data
)
model.fit(X_train, y_train)


# evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
print("\nDetailed Report:")
print(classification_report(y_test, y_pred))

# save the model
# wb means write binary, since we are writing a model file, model_path  is file namne model.pl
with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)


print(f"Model saved to: {MODEL_PATH}")
print("\nReady for Script 3 - Live Detection!")
