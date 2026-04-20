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
