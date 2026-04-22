import cv2  # handles teh webcam, drawing on scrreen, showing the window
import mediapipe as mp
import numpy as np
import pickle  # its used to saved you trained machine learning model into a file
import os  # used to interact with you rcomputes operaitng system
# used to create a fixed length buffer to store recent predictions for smoothing
from collections import deque
import config
from features import extract_features, calculate_angle

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from config import MODEL_PATH, POSE_MODEL_PATH
from features import extract_features, get_joint_angle, get_feedback

# MODEL_PATH = r"C:\Users\LOQ\Desktop\GymForm\lateral_raise_model.pkl"
# # this is pre trained model from the mediapipe
# POSE_MODEL_PATH = "pose_landmarker_full.task"

# --------------- Load trained Ml model ------------
print("Loading model...")
with open(MODEL_PATH, "rb") as f:
    clf = pickle.load(f)
print("Model loaded!")

# ------------ set up MediaPipe ----------------
# tells mediapipe where to find the pose detection model file
base_options = python.BaseOptions(model_asset_path=POSE_MODEL_PATH)
options = vision.PoseLandmarkerOptions(
    base_options=base_options, running_mode=vision.RunningMode.IMAGE, output_segmentation_masks=False)
# configures mediapipe with three settings:
# base_options-the model file location from aboce
# running_mode=IMAGE - process one image at a time(not video stream mode)
# output_segmentation_masks=False - we dont need the body outline mask, just the landmars
detector = vision.PoseLandmarker.create_from_options(options)
# creates teh actual pose detector object using those settings. THIS is what we call later to detect joints
# loads the ai model
# configures it using options
# prepares it to detect poses


prediction_buffer = deque(maxlen=10)  # store last 10 predictions for smoothing
rep_count = 0
rep_stage = None
form_score_correct = 0
form_score_total = 0

# ========================================================================
# def calculate_angle(a, b, c):
#     a = np.array(a)
#     b = np.array(b)
#     c = np.array(c)

#     radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
#         np.arctan2(a[1] - b[1], a[0] - b[0])
#     angle = np.abs(radians * 180.0 / np.pi)

#     if angle > 180.0:
#         angle = 360 - angle

#     return angle
# ========================================================================

CONNECTIONS = [
    (11, 13), (13, 15),  # left arm
    (12, 14), (14, 16),  # right arm
    (11, 12),          # shoulders
    (11, 23), (12, 24),  # torso
    (23, 24),          # hips
    (23, 25), (25, 27),  # left leg
    (24, 26), (26, 28),  # right leg
]
KEY_JOINTS = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]


def draw_skeleton(frame, landmarks, h, w, good_form):
    color = (0, 255, 0) if good_form else (0, 80, 255)
    for s, e in CONNECTIONS:
        x1 = int(landmarks[s].x * w)
        y1 = int(landmarks[s].y * h)

        x2 = int(landmarks[e].x * w)
        y2 = int(landmarks[e].y * h)

        cv2.line(frame, (x1, y1), (x2, y2), color, 2)

    for idx in KEY_JOINTS:
        X = int(landmarks[idx].x * w)
        y = int(landmarks[idx].y * h)
        cv2.circle(frame, (X, y), 5, (0, 255, 255), -1)


def draw_ui(frame, smoothed, conf_pct, feedback_lines, rep_counter, rep_stage, angles, form_score_correct, form_score_total, h, w):

    good_form = smoothed == "correct"
    header_color = (0, 160, 0) if good_form else (0, 0, 180)

    # ----------- header bar -----------------
    cv2.rectangle(frame, (0, 0), (w, 55), header_color, -1)
    if good_form:
        header_text = f"Good Form ({conf_pct:.0f}%)"
    else:
        fault = feedback_lines[0] if feedback_lines else "Bad Form"
        header_text = f"{fault} ({conf_pct:.0f}%)"

    cv2.putText(frame, header_text, (12, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2)

    # ---------- Additional faults below  header -----------
    for i, line in enumerate(feedback_lines[1:], start=1):
        cv2.putText(frame, f"  {line}", (12, 55 + i*26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 220, 60), 2)

    # ------ rep counter ------------
    panel_y = h - 80
    cv2.rectangle(frame, (0, panel_y), (180, h), (30, 30, 30), -1)
    cv2.putText(frame, f"Reps: {rep_count}", (10, panel_y+35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)
    cv2.putText(frame, f"Stage: {rep_stage or '--'}", (10, panel_y+62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

    # -------- quick hint -----------
    cv2.putText(frame, "Q = quit | R = reset reps", (w//2 - 150, h-8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)


# ── Open webcam ───────────────────────────────────────────────────────────────
print("Opening webcam...  Press Q to quit, R to reset rep counter")
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    # Convert to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    # Detect pose
    result = detector.detect(mp_image)

    if result.pose_landmarks and len(result.pose_landmarks) > 0:
        landmarks = result.pose_landmarks[0]

        # ── Get individual joint angles for display and rules ─────────────────
        angles = get_joint_angle(landmarks)

        # ── Rep counter ───────────────────────────────────────────────────────
        avg_angle = (angles["left_shoulder_angle"] +
                     angles["right_shoulder_angle"]) / 2
        if avg_angle < config.REP_DOWN_ANGLE:
            rep_stage = "down"
        if avg_angle > config.REP_UP_ANGLE and rep_stage == "down":
            rep_stage = "up"
            rep_count += 1

        # ── ML prediction ─────────────────────────────────────────────────────
        feat = extract_features(landmarks)
        prediction = clf.predict([feat])[0]
        confidence = clf.predict_proba([feat])[0]
        conf_pct = max(confidence) * 100

        # ── Hard rule override ────────────────────────────────────────────────
        min_elbow = min(angles["left_elbow_angle"],
                        angles["right_elbow_angle"])
        if min_elbow < config.ELBOW_TOO_BENT:
            prediction = "incorrect"

        # ── Smoothing ─────────────────────────────────────────────────────────
        prediction_buffer.append(prediction)
        smoothed = max(set(prediction_buffer),
                       key=list(prediction_buffer).count)

        # ── Form score ────────────────────────────────────────────────────────
        form_score_total += 1
        if smoothed == "correct":
            form_score_correct += 1

        # ── Feedback messages ─────────────────────────────────────────────────
        feedback_lines = get_feedback(angles, config)

        # ── Draw skeleton (green if good, red if bad) ─────────────────────────
        draw_skeleton(frame, landmarks, h, w, smoothed == "correct")

        # ── Shoulder angle numbers ────────────────────────────────────────────
        for lm_idx, angle_key, ox, oy in [
            (11, "left_shoulder_angle",  -55, -10),
            (12, "right_shoulder_angle",  10, -10),
        ]:
            px = int(landmarks[lm_idx].x * w)
            py = int(landmarks[lm_idx].y * h)
            cv2.putText(frame, f"{int(angles[angle_key])}",
                        (px+ox, py+oy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # ── Elbow angle numbers (colour-coded) ────────────────────────────────
        for lm_idx, angle_key, ox, oy in [
            (13, "left_elbow_angle",  -60, 0),
            (14, "right_elbow_angle",  10, 0),
        ]:
            px = int(landmarks[lm_idx].x * w)
            py = int(landmarks[lm_idx].y * h)
            ok = angles[angle_key] >= config.ELBOW_TOO_STRAIGHT
            cv2.putText(frame, f"{int(angles[angle_key])}",
                        (px+ox, py+oy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 255, 0) if ok else (0, 0, 255), 2)

        # ── Draw full UI overlay ──────────────────────────────────────────────
        draw_ui(frame, smoothed, conf_pct, feedback_lines,
                rep_count, rep_stage, angles,
                form_score_correct, form_score_total, h, w)

    else:
        cv2.rectangle(frame, (0, 0), (w, 55), (50, 50, 50), -1)
        cv2.putText(frame, "No person detected — step back", (12, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)

    cv2.imshow("Gym Form Detector", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('r'):
        rep_count = 0
        rep_stage = None
        form_score_correct = 0
        form_score_total = 0
        prediction_buffer.clear()
        print("Rep counter and score reset!")

cap.release()
cv2.destroyAllWindows()

if form_score_total > 0:
    final_score = form_score_correct / form_score_total * 100
    print(f"\nSession Summary")
    print(f"  Reps completed : {rep_count}")
    print(f"  Form score     : {final_score:.1f}%")
    print(f"  Total frames   : {form_score_total}")
print("Done!")
