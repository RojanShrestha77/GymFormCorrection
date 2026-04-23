import numpy as np

from config import N_FEATURES


def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    # arctan2 = is a math function that helps you find the angle of a point from teh origin in a 2D plane.
    # you are calculating the angle at point B formed by three points:
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
        np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180:
        angle = 360 - angle

    return angle


def extract_features(landmarks):
    # --- joints ----
    left_shoulder = [landmarks[11].x, landmarks[11].y]
    right_shoulder = [landmarks[12].x, landmarks[12].y]

    left_elbow = [landmarks[13].x, landmarks[13].y]
    right_elbow = [landmarks[14].x, landmarks[14].y]

    left_hip = [landmarks[23].x, landmarks[23].y]
    right_hip = [landmarks[24].x, landmarks[24].y]

    left_wrist = [landmarks[15].x, landmarks[15].y]
    right_wrist = [landmarks[16].x, landmarks[16].y]

    # angles
    left_shoulder_angle = calculate_angle(left_hip, left_shoulder, left_elbow)
    right_shoulder_angle = calculate_angle(
        right_hip, right_shoulder, right_elbow)

    left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    right_elbow_angle = calculate_angle(
        right_shoulder, right_elbow, right_wrist)

    # smart features
    avg_shoulder_angle = (left_shoulder_angle + right_shoulder_angle) / 2
    avg_elbow_angle = (left_elbow_angle + right_elbow_angle) / 2
    symmetry = abs(left_shoulder_angle - right_shoulder_angle)

    # final feature vector
    feat = [
        avg_shoulder_angle,
        avg_elbow_angle,
        symmetry
    ]

    # raw landmarks
    for lm in landmarks:
        feat += [lm.x, lm.y, lm.z, lm.visibility]

    # safety check - catches mismatch before it causes silent bugs
    # I expect len(feat) to be equal to N_FEATURES.
    # if not stop the program and show the below message
    assert len(feat) == N_FEATURES, (
        f"Feature length mismatch: got {len(feat)}, expected {N_FEATURES}. "
        f"Check config.py N_FEATURES and extract_features() stay in sync."
    )

    return feat


def get_joint_angle(landmarks):
    """
    Returns indivudal joint angles seperately.
    Used bylive_detection.py for displaying angles on screen and rule based checks.
    """
    left_shoulder = [landmarks[11].x, landmarks[11].y]
    right_shoulder = [landmarks[12].x, landmarks[12].y]
    left_elbow = [landmarks[13].x, landmarks[13].y]
    right_elbow = [landmarks[14].x, landmarks[14].y]
    left_wrist = [landmarks[15].x, landmarks[15].y]
    right_wrist = [landmarks[16].x, landmarks[16].y]
    left_hip = [landmarks[23].x, landmarks[23].y]
    right_hip = [landmarks[24].x, landmarks[24].y]
    # x = horizontal position in the image (0->1) left right position of the image in the screen
    # y = vertical position in the image (0->1) up down positon of the image(eg: person in our case) in the screen

    return {
        # it finds the angle between a b and c where b is the vertex or center.
        "left_shoulder_angle": calculate_angle(left_hip, left_shoulder, left_elbow),
        "right_shoulder_angle": calculate_angle(right_hip, right_shoulder, right_elbow),
        "left_elbow_angle": calculate_angle(left_shoulder, left_elbow, left_wrist),
        "right_elbow_angle": calculate_angle(right_shoulder, right_elbow, right_wrist)
    }


def get_feedback(angles: dict, config) -> list:
    """
    Rule-based feedback messages from joint angles.
    Returns a list of strings — empty list means form looks fine.
    """
    feedback = []

    min_elbow = min(angles["left_elbow_angle"], angles["right_elbow_angle"])
    min_shoulder = min(angles["left_shoulder_angle"],
                       angles["right_shoulder_angle"])
    max_shoulder = max(angles["left_shoulder_angle"],
                       angles["right_shoulder_angle"])
    symmetry = abs(angles["left_shoulder_angle"] -
                   angles["right_shoulder_angle"])

    if min_elbow < config.ELBOW_TOO_BENT:
        feedback.append(f"Straighten elbows ({int(min_elbow)}°)")

    if min_shoulder < config.ARM_TOO_LOW:
        feedback.append(f"Raise arms higher ({int(min_shoulder)}°)")

    if max_shoulder > config.ARM_TOO_HIGH:
        feedback.append(f"Don't raise arms too high ({int(max_shoulder)}°)")

    if symmetry > config.SYMMETRY_MAX:
        feedback.append(f"Keep arms even ({int(symmetry)}° difference)")

    return feedback
