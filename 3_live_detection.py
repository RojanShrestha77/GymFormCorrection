import cv2  # handles teh webcam, drawing on scrreen, showing the window
import mediapipe as mp
import numpy as np
import pickle  # its used to saved you trained machine learning model into a file
import os  # used to interact with you rcomputes operaitng system
# used to create a fixed length buffer to store recent predictions for smoothing
from collections import deque


from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = r"C:\Users\LOQ\Desktop\GymForm\lateral_raise_model.pkl"
# this is pre trained model from the mediapipe
POSE_MODEL_PATH = "pose_landmarker_full.task"

# Load trained Ml model ---
print("Loading model...")
with open(MODEL_PATH, "rb") as f:
    clf = pickle.load(f)
print("Model loaded!")

# --- set up MediaPipe ----
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


def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
        np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


def draw_skeleton(frame, landmarks, h, w):
    # ddraw skeletion lines and joints dots on the fram
    connections = [
        (11, 13), (13, 15),  # left arm
        (12, 14), (14, 16),  # right arm
        (11, 12),  # shoulders
        (11, 23), (12, 24),  # torso sides
        (23, 24),  # hips
        (23, 25), (25, 27),  # left leg
        (24, 26), (26, 28)  # right leg
    ]

    # DRaw lines
    for start, end in connections:
        # landmarks = key body points detected by AI
        x1 = int(landmarks[start].x * w)
        y1 = int(landmarks[start].y * h)
        x2 = int(landmarks[end].x * w)
        y2 = int(landmarks[end].y * h)
        cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
        # frame, (x1, y1), (x2, y2), draws lines between two points and connects them
        # color = (255, 255, 255), thickness - 2

    # draw dots on joints
    key_joints = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

    # draws green dots on important body joints (liek shoulders, ,elbows, knees, etc)
    for idx in key_joints:
        x = int(landmarks[idx].x * w)
        y = int(landmarks[idx].y * h)
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

# specific feedback messages


def get_feedback(left_angle, right_angle, left_elbow_angle, right_elbow_angle):
    """Returns a list of specific form fault messages."""
    feedback = []
    if left_elbow_angle < 150 or right_elbow_angle < 150:
        feedback.append(
            f"Straighten elbows ({int(min(left_elbow_angle, right_elbow_angle))}deg)")
    if left_angle < 60 or right_angle < 60:
        feedback.append("Raise arms higher")
    if left_angle > 100 or right_angle > 100:
        feedback.append("Don't go above parallel")
    if abs(left_angle - right_angle) > 20:
        feedback.append("Keep arms even")
    return feedback


rep_count = 0
rep_stage = None

# ------ open webcan ---
print("Opening webcam------- Press Q to quit")
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    # convert to rgb for mediapipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    # Detect pose
    result = detector.detect(mp_image)

    if result.pose_landmarks and len(result.pose_landmarks) > 0:
        landmarks = result.pose_landmarks[0]

        # Draw skeleton
        draw_skeleton(frame, landmarks, h, w)

        # --- calculate arm angles ---
        # left arm: hip(23) -> shoulder(11) -> elbow(13)
        left_hip = [landmarks[23].x, landmarks[23].y]
        left_shoulder = [landmarks[11].x, landmarks[11].y]
        left_elbow = [landmarks[13].x, landmarks[13].y]
        left_angle = calculate_angle(left_hip, left_shoulder, left_elbow)

        # Right arm: hip(24) -> shoulder(12) -> elbow(14)
        right_hip = [landmarks[24].x, landmarks[24].y]
        right_shoulder = [landmarks[12].x, landmarks[12].y]
        right_elbow = [landmarks[14].x, landmarks[14].y]
        right_angle = calculate_angle(right_hip, right_shoulder, right_elbow)

        # rep counter
        avg_angle = (left_angle + right_angle) / 2
        if avg_angle < 30:
            rep_stage = "down"
        if avg_angle > 70 and rep_stage == "down":
            rep_stage = "up"
            rep_count += 1

        # elbow bend angle
        left_wrist = [landmarks[15].x, landmarks[15].y]
        right_wrist = [landmarks[16].x, landmarks[16].y]

        # a = shoulder, b = elbow(center), c = wrist
        left_elbow_angle = calculate_angle(
            left_shoulder, left_elbow, left_wrist)
        right_elbow_angle = calculate_angle(
            right_shoulder, right_elbow, right_wrist)

        # --- Prepare features for ML model ---
        feat = []
        for lm in landmarks:
            feat += [lm.x, lm.y, lm.z, lm.visibility]

        prediction = clf.predict([feat])[0]
        confidence = clf.predict_proba([feat])[0]
        conf_pct = max(confidence) * 100

        # hard rule: if eblows are bent, override ml prediction
        # degrees - straigh arm is 180, so anything less than 150 is probably a bent arm
        elbow_threshold = 150
        if left_elbow_angle < elbow_threshold or right_elbow_angle < elbow_threshold:
            prediction = "incorrect"
            label = f"Elbows  bent! ({int(min(left_elbow_angle, right_elbow_angle))}°)"

        prediction_buffer.append(prediction)
        smoothed = max(set(prediction_buffer),
                       key=list(prediction_buffer).count)

        #  specific feedback messages ───────────────────────────
        feedback_lines = get_feedback(left_angle, right_angle,
                                      left_elbow_angle, right_elbow_angle)

        # ── Display: header bar ──────────────────────────────────────────────
        if smoothed == "correct":
            label = f"Good Form ({conf_pct:.0f}%)"
            bg = (0, 160, 0)
        else:
            # Show first fault in the header if we have one, else generic
            fault = feedback_lines[0] if feedback_lines else "Bad Form"
            label = f"{fault} ({conf_pct:.0f}%)"
            bg = (0, 0, 180)

        cv2.rectangle(frame, (0, 0), (w, 50), bg, -1)
        cv2.putText(frame, label, (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # ── FEATURE 4: extra fault lines below header ────────────────────────
        # show remaining faults (skip [0], already in header)
        for i, line in enumerate(feedback_lines[1:], start=1):
            cv2.putText(frame, f"  {line}",
                        (10, 50 + i * 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 220, 80), 2)

        # Display angles on screen
        lx = int(landmarks[11].x * w)
        ly = int(landmarks[11].y * h)
        rx = int(landmarks[12].x * w)
        ry = int(landmarks[12].y * h)
        ex_l = int(landmarks[13].x * w)
        ey_l = int(landmarks[13].y * h)
        ex_r = int(landmarks[14].x * w)
        ey_r = int(landmarks[14].y * h)

        cv2.putText(frame, f"{int(left_angle)}",
                    (lx - 40, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.putText(frame, f"{int(right_angle)}",
                    (rx + 10, ry),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # elbow bend angle (new — shown at elbow joint)
        elbow_color_l = (0, 255, 0) if left_elbow_angle >= 150 else (0, 0, 255)
        elbow_color_r = (
            0, 255, 0) if right_elbow_angle >= 150 else (0, 0, 255)
        cv2.putText(frame, f"{int(left_elbow_angle)}deg",  (ex_l - 50, ey_l),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, elbow_color_l, 2)
        cv2.putText(frame, f"{int(right_elbow_angle)}deg", (ex_r + 10, ey_r),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, elbow_color_r, 2)

        # Rep counter display
        cv2.putText(frame, f"Reps: {rep_count}",
                    (10, h - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        cv2.putText(frame, f"Stage: {rep_stage or '--'}",
                    (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    else:
        # No person detected
        cv2.rectangle(frame, (0, 0), (w, 50), (50, 50, 50), -1)
        cv2.putText(frame, "No person detected - step back",
                    (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Lateral Raise Form Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
print("Done!")
