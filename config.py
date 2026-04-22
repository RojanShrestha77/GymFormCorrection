# this is the single source of truth for the whole project
# every script imports from here. Change once, affects everything.

import os

# ----- paths --------
BASE_DIR = r"C:\Users\LOQ\Desktop\GymForm"
CSV_PATH = os.path.join(BASE_DIR, "lateral_raise_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "lateral_raise_model.pkl")
POSE_MODEL_PATH = os.path.join(BASE_DIR, "pose_landmarker_full.task")


# ------ feature deinition -------
# this is teh single place that defines what features exist, if you add a new feature , add it here only - nowhere else

ANGLE_FEATURES = ["avg_shoulder_angle", "avg_elbow_angle", "symmetry"]
N_LANDMARKS = 33
N_RAW = N_LANDMARKS * 4  # x, y, z visibility per landmark
N_ANGLE = len(ANGLE_FEATURES)
N_FEATURES = N_ANGLE + N_RAW

# column names for the csv files
RAW_COLS = []
for i in range(N_LANDMARKS):
    RAW_COLS += [f"x_{i}", f"y_{i}", f"z_{i}", f"v_{i}"]

ALL_FEATURE_COLS = ANGLE_FEATURES + RAW_COLS

CSV_COLUMNS = ALL_FEATURE_COLS + ["label"]

# --- model settings -----
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_ESTIMATORS = 200
CV_FOLDS = 5

# -------- Rep counter thresholds ----
REP_DOWN_ANGLE = 30     # arm considerred "down" below this angle
REP_UP_ANGLE = 70       # arm considered up above this angle

# ------- form thresholds -------
ELBOW_TOO_BENT = 150       # below this = too bedn = bad form
ELBOW_TOO_STRAIGHT = 175   # above this = too straight = bad form
ARM_TOO_LOW = 60           # below this at top = not raised enough
ARM_TOO_HIGH = 100         # above this = raised too high
SYMMETRY_MAX = 20           # arm angle difference above this = uneven

# ------- smoothin ------------
PREDICTION_BUFFER_SIZE = 10  # frames to average prediction over
