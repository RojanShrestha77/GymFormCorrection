import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
from features import extract_features

# New MediaPipe API
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request

# -----------------------------------------------
CORRECT_FOLDER = r"C:\Users\LOQ\Desktop\GymForm\laterraises-dataset\lateral raises"
INCORRECT_FOLDER = r"C:\Users\LOQ\Desktop\GymForm\laterraises-dataset\lateral-raise.multiclass\train"
OUTPUT_CSV = r"C:\Users\LOQ\Desktop\GymForm\lateral_raise_data.csv"
MODEL_PATH = "pose_landmarker_full.task"
# -----------------------------------------------

# Download the model file if not already present
if not os.path.exists(MODEL_PATH):
    print("Downloading MediaPipe pose model...")
    url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
    urllib.request.urlretrieve(url, MODEL_PATH)
    print("Model downloaded!")

# Set up the pose landmarker
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE
)
detector = vision.PoseLandmarker.create_from_options(options)


# def get_landmarks(image_path):
#     """Run MediaPipe on one image and return landmarks as a flat list"""
#     image = cv2.imread(image_path)
#     if image is None:
#         return None

#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#     mp_image = mp.Image(
#         image_format=mp.ImageFormat.SRGB,
#         data=image_rgb
#     )

#     result = detector.detect(mp_image)

#     if not result.pose_landmarks or len(result.pose_landmarks) == 0:
#         return None

#     landmarks = []
#     for landmark in result.pose_landmarks[0]:
#         landmarks.append(landmark.x)
#         landmarks.append(landmark.y)
#         landmarks.append(landmark.z)
#         landmarks.append(landmark.visibility)

#     return landmarks  # 33 landmarks x 4 values = 132 numbers

def get_landmarks(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    result = detector.detect(mp_image)

    if not result.pose_landmarks or len(result.pose_landmarks) == 0:
        return None

    landmarks = result.pose_landmarks[0]  # ✅ keep as object, not flat list
    return extract_features(landmarks)    # ✅ returns 135 features


def process_folder(folder_path, label):
    """Process all images in a folder and return rows for the CSV"""
    rows = []
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        return rows

    files = [f for f in os.listdir(folder_path)
             if os.path.splitext(f)[1].lower() in image_extensions]

    print(f"\nProcessing '{label}' images from: {folder_path}")
    print(f"Found {len(files)} images")

    success = 0
    failed = 0

    for filename in files:
        image_path = os.path.join(folder_path, filename)
        landmarks = get_landmarks(image_path)

        if landmarks is not None:
            rows.append(landmarks + [label])
            success += 1
        else:
            failed += 1
            print(f"  Skipped (no person detected): {filename}")

    print(f"  Successfully processed: {success}")
    print(f"  Skipped (no person found): {failed}")
    return rows


# # --- Build column names ---
# columns = []
# for i in range(33):
#     columns += [f"x{i}", f"y{i}", f"z{i}", f"v{i}"]
# columns.append("label")

columns = ["avg_shoulder_angle", "avg_elbow_angle",
           "symmetry"]  # 3 angle features
for i in range(33):
    columns += [f"x{i}", f"y{i}", f"z{i}", f"v{i}"]  # 132 raw landmarks
columns.append("label")
# total = 135 + 1

# --- Process both folders ---
all_rows = []
all_rows += process_folder(CORRECT_FOLDER,   label="correct")
all_rows += process_folder(INCORRECT_FOLDER, label="incorrect")

# --- Save to CSV ---
if len(all_rows) == 0:
    print("\nNo data collected! Check your folder paths.")
else:
    df = pd.DataFrame(all_rows, columns=columns)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nDone! Saved {len(df)} rows to '{OUTPUT_CSV}'")
    print(f"  Correct samples:   {len(df[df['label'] == 'correct'])}")
    print(f"  Incorrect samples: {len(df[df['label'] == 'incorrect'])}")
