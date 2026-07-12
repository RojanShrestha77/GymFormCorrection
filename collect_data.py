"""
collect_data.py - Collect TCN training data for lateral raise

HOW IT WORKS:
  Press S to START recording a rep.
  Perform the full rep (raise and lower).
  Press S again to SAVE the rep.
  Each saved rep = 1 CSV row = 30 frames x 6 features = 180 values.

WHAT TO COLLECT (aim for these minimums):
  correct          -- 50+ reps, good form from start to finish
  elbow_bent       -- 30+ reps, keep elbows bent throughout raise
  not_high_enough  -- 30+ reps, only raise arms halfway
  torso_lean       -- 30+ reps, lean sideways as you raise

CONTROLS:
  S = Start recording / Save recorded rep
  Q = Quit
"""

import pandas as pd
import cv2
import csv
import os
import mediapipe as mp
from collections import deque

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from features import (
    extract_frame_features,
    build_csv_header,
    get_rule_based_feedback,
    _check_visibility,
)
from config import (
    POSE_MODEL_PATH,
    CSV_PATH,
    SEQ_LEN,
    N_FEATURES,
    LABELS,
    get_logger,
)

logger = get_logger("collect_data")


# ── MediaPipe setup (same as your existing scripts) ───────────────────
logger.info("Setting up MediaPipe pose detector...")
base_options = python.BaseOptions(model_asset_path=POSE_MODEL_PATH)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
)
detector = vision.PoseLandmarker.create_from_options(options)
logger.info("MediaPipe ready.")


# ── Get label from user ───────────────────────────────────────────────
print("\nAvailable labels:")
for i, lbl in enumerate(LABELS):
    print(f"  {i+1}. {lbl}")

raw = input("\nEnter label number or name: ").strip()

if raw.isdigit():
    idx = int(raw) - 1
    if 0 <= idx < len(LABELS):
        label = LABELS[idx]
    else:
        print("Invalid number.")
        exit(1)
elif raw in LABELS:
    label = raw
else:
    print(f"Invalid. Choose from: {LABELS}")
    exit(1)

logger.info(f"Collecting label: '{label}'")


# ── CSV setup ─────────────────────────────────────────────────────────
file_exists = os.path.exists(CSV_PATH)
csv_file = open(CSV_PATH, "a", newline="")
writer = csv.writer(csv_file)

if not file_exists:
    writer.writerow(build_csv_header())
    logger.info(f"Created new CSV: {CSV_PATH}")
else:
    logger.info(f"Appending to existing CSV: {CSV_PATH}")


# ── State ─────────────────────────────────────────────────────────────
frame_buffer = []       # list of feature vectors for current rep
prev_shoulder_angle = None
is_recording = False
saved_reps = 0
s_pressed_last = False    # debounce S key


# ── Progress bar helper ───────────────────────────────────────────────
def draw_bar(frame, filled, total, x, y, w, h):
    cv2.rectangle(frame, (x, y), (x+w, y+h), (80, 80, 80), 2)
    if total > 0:
        fw = int(w * min(filled, total) / total)
        color = (0, 200, 255) if filled >= total else (0, 255, 100)
        if fw > 0:
            cv2.rectangle(frame, (x, y), (x+fw, y+h), color, -1)


# ── Webcam ────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    logger.error("Cannot open webcam.")
    exit(1)

logger.info("Ready. Press S to start/save a rep. Press Q to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    H, W = frame.shape[:2]

    # ── Pose detection ────────────────────────────────────────────────
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)

    pose_ok = (
        result.pose_landmarks
        and len(result.pose_landmarks) > 0
        and _check_visibility(result.pose_landmarks[0])
    )

    # ── Draw skeleton dots ────────────────────────────────────────────
    if pose_ok:
        for lm in result.pose_landmarks[0]:
            cx, cy = int(lm.x * W), int(lm.y * H)
            cv2.circle(frame, (cx, cy), 3, (0, 255, 120), -1)

    # ── Key handling ─────────────────────────────────────────────────
    key = cv2.waitKey(1) & 0xFF

    s_now = (key == ord('s'))
    if s_now and not s_pressed_last:
        # Toggle recording on fresh S press
        if not is_recording:
            # Start
            is_recording = True
            frame_buffer = []
            prev_shoulder_angle = None
            logger.info("Recording started...")
        else:
            # Stop and attempt save
            is_recording = False
            if len(frame_buffer) >= SEQ_LEN:
                rep_data = frame_buffer[:SEQ_LEN]  # trim to exactly SEQ_LEN
                flat = [v for fv in rep_data for v in fv]
                writer.writerow([label] + flat)
                csv_file.flush()
                saved_reps += 1
                logger.info(
                    f"Saved rep #{saved_reps}  ({len(frame_buffer)} frames buffered)")
            else:
                logger.warning(
                    f"Rep too short ({len(frame_buffer)}/{SEQ_LEN} frames). "
                    "Perform the full rep before pressing S."
                )
            frame_buffer = []
    s_pressed_last = s_now

    # ── Buffer frames while recording ────────────────────────────────
    if is_recording and pose_ok:
        lms = result.pose_landmarks[0]
        feats, prev_shoulder_angle = extract_frame_features(
            lms, prev_shoulder_angle)
        frame_buffer.append(feats)

        if len(frame_buffer) == SEQ_LEN:
            logger.info(f"Buffer full ({SEQ_LEN} frames). Press S to save.")

    # ── UI ────────────────────────────────────────────────────────────
    # Dark top bar
    cv2.rectangle(frame, (0, 0), (W, 95), (20, 20, 20), -1)

    # Label + saved count
    cv2.putText(frame, f"Label: {label}", (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 230, 180), 2)
    cv2.putText(frame, f"Saved: {saved_reps} reps", (10, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

    # Recording indicator (top right)
    if is_recording:
        cv2.circle(frame, (W - 24, 20), 10, (0, 0, 220), -1)
        rec_txt = f"REC  {len(frame_buffer)}/{SEQ_LEN}"
        cv2.putText(frame, rec_txt, (W - 180, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 80, 255), 2)
    else:
        cv2.putText(frame, "READY", (W - 100, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)

    # Frame progress bar
    draw_bar(frame, len(frame_buffer), SEQ_LEN, 10, 65, W - 20, 16)

    # Rule-based form hints (shown live while recording)
    if pose_ok and is_recording:
        hints = get_rule_based_feedback(result.pose_landmarks[0])
        for i, hint in enumerate(hints[:2]):  # max 2 hints on screen
            cv2.putText(frame, hint, (10, H - 40 + i*22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 100, 255), 2)

    # No pose warning
    if not pose_ok:
        cv2.putText(frame, "No pose detected — step back / check lighting",
                    (10, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 220), 2)

    # Controls hint (bottom right)
    cv2.putText(frame, "S = Start/Save rep    Q = Quit",
                (W - 310, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)

    cv2.imshow(f"GymForm Data Collection  [{label}]", frame)

    if key == ord('q'):
        logger.info(f"Quit. Reps saved this session: {saved_reps}")
        break


# ── Cleanup ───────────────────────────────────────────────────────────
cap.release()
csv_file.close()
cv2.destroyAllWindows()


# ── Dataset summary ───────────────────────────────────────────────────
if os.path.exists(CSV_PATH):
    df = pd.read_csv(CSV_PATH)
    print(f"\n{'='*50}")
    print(f"Dataset: {CSV_PATH}")
    print(f"Total reps: {len(df)}")
    print(
        f"Columns: 1 label + {len(df.columns)-1} features ({SEQ_LEN} frames x {N_FEATURES} features)")
    print(f"\nLabel distribution:")
    print(df['label'].value_counts().to_string())
    print(f"\nTarget: 50 correct + 30 each error type = 140 reps minimum")
    still_needed = max(0, 140 - len(df))
    print(f"Still needed: ~{still_needed} more reps")
    print(f"{'='*50}")
