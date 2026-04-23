import cv2
import csv
import mediapipe as mp
from features import extract_features
from config import POSE_MODEL_PATH, get_logger

# New MediaPipe API
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Set up logging
logger = get_logger("collect_data")

# Set up the pose landmarker (consistent with 1_extract_landmarks.py)
logger.info("Setting up MediaPipe pose detector...")
base_options = python.BaseOptions(model_asset_path=POSE_MODEL_PATH)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE
)
detector = vision.PoseLandmarker.create_from_options(options)
logger.info("MediaPipe setup complete!")

CSV_PATH = "lateral_raise_dataset_v2.csv"

# create file + header
header = ["label"] + [f"f{i}" for i in range(3 + 33*4)]

with open(CSV_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)

logger.info(f"Created CSV file: {CSV_PATH}")

# start webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    logger.error("Failed to open webcam")
    exit(1)

# asking user for label
label = input("Enter label (correct/incorrect): ")
logger.info(f"Collecting samples for label: {label}")
logger.info("Press 'S' to save a sample, 'Q' to quit")

sample_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    
    # Detect pose using new API
    result = detector.detect(mp_image)

    if result.pose_landmarks and len(result.pose_landmarks) > 0:
        landmarks = result.pose_landmarks[0]  # Get first person's landmarks

        cv2.putText(frame, f"Samples: {sample_count} | S=Save | Q=Quit", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Label: {label}", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "No person detected", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Collecting Data", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s') and result.pose_landmarks and len(result.pose_landmarks) > 0:
        try:
            feat = extract_features(landmarks)
            with open(CSV_PATH, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([label] + feat)
            sample_count += 1
            logger.info(f"Saved sample #{sample_count}")
        except Exception as e:
            logger.error(f"Failed to save sample: {e}")

    if key == ord('q'):
        logger.info(f"Collection stopped. Total samples saved: {sample_count}")
        break

cap.release()
cv2.destroyAllWindows()
