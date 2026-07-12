"""
features.py - Biomechanical feature extraction for lateral raise TCN model

WHAT CHANGED FROM OLD VERSION:
  Old: 3 angles + 33*4 raw landmarks = 135 features per FRAME
  New: 6 biomechanical angles = 6 features per FRAME
       30 frames stacked = 180 features per REP (one CSV row)

WHY:
  Raw x,y,z break when you move closer/further from camera.
  Angles are camera-distance independent — 90 degrees is 90 degrees anywhere.

FEATURES PER FRAME:
  0: shoulder_abduction  -- raise angle at shoulder (wrist-shoulder-hip)
  1: elbow_angle         -- how bent the elbow is (shoulder-elbow-wrist)
  2: torso_lean          -- lateral body lean (shoulder-hip-knee)
  3: wrist_height        -- normalized wrist height vs shoulder
  4: angular_velocity    -- how fast shoulder angle is changing
  5: symmetry            -- difference between left and right shoulder angles
"""

import math
from config import N_FEATURES, SEQ_LEN, MIN_VISIBILITY, get_logger

logger = get_logger("features")


# MediaPipe landmark indices (same 33 as before)
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26


def _angle(a, b, c) -> float:
    """
    Angle at joint B between points A-B-C.
    Accepts landmarks with .x and .y attributes (MediaPipe Tasks API format).
    Returns degrees 0-180.
    """
    ba = [a.x - b.x, a.y - b.y]
    bc = [c.x - b.x, c.y - b.y]

    mag_ba = math.sqrt(ba[0]**2 + ba[1]**2)
    mag_bc = math.sqrt(bc[0]**2 + bc[1]**2)

    if mag_ba < 1e-6 or mag_bc < 1e-6:
        return 0.0

    dot = ba[0]*bc[0] + ba[1]*bc[1]
    cosine = dot / (mag_ba * mag_bc)
    cosine = max(-1.0, min(1.0, cosine))
    return math.degrees(math.acos(cosine))


def _check_visibility(landmarks) -> bool:
    """Returns False if any critical joint is occluded or low confidence."""
    key_indices = [
        LEFT_SHOULDER, RIGHT_SHOULDER,
        LEFT_ELBOW, RIGHT_ELBOW,
        LEFT_WRIST, RIGHT_WRIST,
        LEFT_HIP, LEFT_KNEE,
    ]
    for idx in key_indices:
        if landmarks[idx].visibility < MIN_VISIBILITY:
            return False
    return True


def extract_frame_features(landmarks, prev_shoulder_angle: float = None):
    """
    Extract 6 biomechanical features from a single frame.

    Args:
        landmarks:            list of 33 MediaPipe landmarks (Tasks API)
        prev_shoulder_angle:  shoulder angle from previous frame for velocity

    Returns:
        (features, shoulder_angle)
        features:       list of 6 floats
        shoulder_angle: float, pass back in on the next frame call
    """
    lm = landmarks

    # 1. Shoulder abduction (wrist-shoulder-hip)
    # At rest ~0-20 degrees, correct raise ~80-100 degrees
    shoulder_angle = _angle(lm[LEFT_WRIST], lm[LEFT_SHOULDER], lm[LEFT_HIP])

    # 2. Elbow angle (shoulder-elbow-wrist)
    # Should stay ~160-170 degrees (slight bend)
    # Error "elbow_bent" = below 140 degrees
    elbow_angle = _angle(lm[LEFT_SHOULDER], lm[LEFT_ELBOW], lm[LEFT_WRIST])

    # 3. Torso lateral lean (shoulder-hip-knee)
    # Upright = ~175-180 degrees, leaning error = below 165 degrees
    torso_angle = _angle(lm[LEFT_SHOULDER], lm[LEFT_HIP], lm[LEFT_KNEE])

    # 4. Wrist height normalized by body height
    # Positive = wrist above shoulder (correct), Negative = not raised enough
    shoulder_y = lm[LEFT_SHOULDER].y
    wrist_y = lm[LEFT_WRIST].y
    hip_y = lm[LEFT_HIP].y
    body_height = abs(shoulder_y - hip_y) + 1e-6
    # MediaPipe y-axis: 0=top, 1=bottom so subtract to get "up" as positive
    wrist_height = (shoulder_y - wrist_y) / body_height

    # 5. Angular velocity (change in shoulder angle from last frame)
    # Positive = raising (concentric), Negative = lowering (eccentric)
    velocity = (shoulder_angle -
                prev_shoulder_angle) if prev_shoulder_angle is not None else 0.0

    # 6. Left-right symmetry (difference between both shoulder angles)
    right_shoulder_angle = _angle(
        lm[RIGHT_WRIST], lm[RIGHT_SHOULDER], lm[RIGHT_HIP])
    symmetry = abs(shoulder_angle - right_shoulder_angle)

    features = [
        round(shoulder_angle, 4),
        round(elbow_angle,    4),
        round(torso_angle,    4),
        round(wrist_height,   6),
        round(velocity,       4),
        round(symmetry,       4),
    ]

    assert len(features) == N_FEATURES, (
        f"Feature length mismatch: got {len(features)}, expected {N_FEATURES}. "
        f"Check config.py N_FEATURES."
    )

    return features, shoulder_angle


def build_csv_header() -> list:
    """
    Column names for the TCN dataset CSV.
    Format: label, f0_0, f0_1, ..., f29_5
    One row = one full rep = 30 frames x 6 features = 180 feature columns.
    """
    cols = ["label"]
    for frame in range(SEQ_LEN):
        for feat_idx in range(N_FEATURES):
            cols.append(f"f{frame}_{feat_idx}")
    return cols


def get_rule_based_feedback(landmarks) -> list:
    """
    Rule-based form checks for live display while recording.
    Returns list of error strings. Empty list = form looks fine.
    Keeps the same logic as your old get_feedback() function.
    """
    lm = landmarks
    shoulder_angle = _angle(lm[LEFT_WRIST], lm[LEFT_SHOULDER], lm[LEFT_HIP])
    elbow_angle = _angle(lm[LEFT_SHOULDER], lm[LEFT_ELBOW], lm[LEFT_WRIST])
    right_angle = _angle(lm[RIGHT_WRIST], lm[RIGHT_SHOULDER], lm[RIGHT_HIP])
    symmetry = abs(shoulder_angle - right_angle)

    from config import ELBOW_TOO_BENT, ARM_TOO_LOW, ARM_TOO_HIGH, SYMMETRY_MAX
    feedback = []

    if elbow_angle < ELBOW_TOO_BENT:
        feedback.append(f"Straighten elbows ({int(elbow_angle)} deg)")
    if shoulder_angle < ARM_TOO_LOW:
        feedback.append(f"Raise arms higher ({int(shoulder_angle)} deg)")
    if shoulder_angle > ARM_TOO_HIGH:
        feedback.append(f"Don't raise too high ({int(shoulder_angle)} deg)")
    if symmetry > SYMMETRY_MAX:
        feedback.append(f"Keep arms even ({int(symmetry)} deg diff)")

    return feedback
