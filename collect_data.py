import cv2
import csv
import mediapipe as mp
from features import extract_features

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()  # Fixed: Pose() not pose()

CSV_PATH = "lateral_raise_dataset_v2.csv"

# create file + header
header = ["label"] + [f"f{i}" for i in range(3 + 33*4)]

with open(CSV_PATH, "w", newline="") as f:
    writer = csv.writer(f)   # Fixed: writer() not write()
    writer.writerow(header)

# start webcam
cap = cv2.VideoCapture(0)

# asking user for label
label = input("Enter label (correct/incorrect): ")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Fixed: cvtColor not cvtColot
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)

    if result.pose_landmarks:
        landmarks = result.pose_landmarks.landmark

        cv2.imshow("Collecting Data", frame)

        key = cv2.waitKey(1) & 0xFF  # Fixed: waitKey not waitkey

        if key == ord('s'):
            feat = extract_features(landmarks)

            with open(CSV_PATH, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([label] + feat)

            print("Saved sample!")

        cv2.imshow("Collecting Data", frame)  # ✅ always show frame
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
