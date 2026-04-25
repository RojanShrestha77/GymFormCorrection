import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
from features import extract_features
from config import CORRECT_FOLDER, INCORRECT_FOLDER, POSE_MODEL_PATH, CSV_PATH, get_logger

# New MediaPipe API
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request

# Set up logging
logger = get_logger("extract_landmarks")

# Download the model file if not already present
if not os.path.exists(POSE_MODEL_PATH):
    logger.info("Downloading MediaPipe pose model...")
    try:
        url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
        urllib.request.urlretrieve(url, POSE_MODEL_PATH)
        logger.info("Model downloaded successfully!")
    except Exception as e:
        logger.error(f"Failed to download MediaPipe model: {e}")
        logger.error("Check your internet connection and try again.")
        exit(1)
else:
    logger.info("MediaPipe model already exists")

# Set up the pose landmarker
logger.info("Setting up MediaPipe pose detector...")
base_options = python.BaseOptions(model_asset_path=POSE_MODEL_PATH)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE
)
detector = vision.PoseLandmarker.create_from_options(options)
logger.info("MediaPipe setup complete!")


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
        logger.warning(f"Folder not found: {folder_path}")
        return rows

    files = [f for f in os.listdir(folder_path)
             if os.path.splitext(f)[1].lower() in image_extensions]

    logger.info(f"\nProcessing '{label}' images from: {folder_path}")
    logger.info(f"Found {len(files)} images")

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
            logger.debug(f"  Skipped (no person detected): {filename}")

    logger.info(f"  Successfully processed: {success}")
    logger.info(f"  Skipped (no person found): {failed}")
    return rows


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

# --- Save to CSV with error handling ---
if len(all_rows) == 0:
    logger.error("\nNo data collected! Check your folder paths.")
    logger.error(f"  CORRECT_FOLDER: {CORRECT_FOLDER}")
    logger.error(f"  INCORRECT_FOLDER: {INCORRECT_FOLDER}")
else:
    df = pd.DataFrame(all_rows, columns=columns)
    try:
        df.to_csv(CSV_PATH, index=False)
        logger.info(f"\nDone! Saved {len(df)} rows to '{CSV_PATH}'")
        logger.info(
            f"  Correct samples:   {len(df[df['label'] == 'correct'])}")
        logger.info(
            f"  Incorrect samples: {len(df[df['label'] == 'incorrect'])}")
    except PermissionError:
        logger.error(f"Permission denied: Cannot write to {CSV_PATH}")
        exit(1)
    except Exception as e:
        logger.error(f"Failed to save CSV: {e}")
        exit(1)
