# config.py - Single source of truth for the GymForm TCN pipeline
# Updated for new architecture: per-rep CSV + TCN model + TFLite export
# Keep your BASE_DIR and logging — everything else updated for TCN

import os
import logging
from datetime import datetime

# ── Paths ─────────────────────────────────────────────────────────────
BASE_DIR = r"C:\Users\LOQ\Desktop\GymForm"
POSE_MODEL_PATH = os.path.join(BASE_DIR, "pose_landmarker_full.task")

# NEW: TCN dataset (per-rep, not per-frame)
CSV_PATH = os.path.join(BASE_DIR, "lateral_raise_tcn_augmented.csv")

# NEW: TFLite model output (replaces .pkl)
TFLITE_MODEL_PATH = os.path.join(BASE_DIR, "lateral_raise_model.tflite")
KERAS_MODEL_PATH = os.path.join(BASE_DIR, "lateral_raise_model.keras")

# ── Sequence settings ─────────────────────────────────────────────────
SEQ_LEN = 30   # frames per rep
N_FEATURES = 6    # biomechanical features per frame (see features.py)
# Total columns per row in CSV = SEQ_LEN * N_FEATURES = 180

# ── Labels ────────────────────────────────────────────────────────────
# 4 classes — must match what you collect in collect_data.py
LABELS = ["correct", "elbow_bent", "not_high_enough", "torso_lean"]
N_CLASSES = len(LABELS)  # 4

# ── Model hyperparameters ─────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.2
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 15
CV_FOLDS = 5

# ── Rep counter thresholds (rule-based, used in collect_data.py UI) ──
REP_DOWN_ANGLE = 30   # shoulder angle below this = arm down
REP_UP_ANGLE = 70   # shoulder angle above this = arm up

# ── Form thresholds (used for rule-based checks alongside model) ──────
ELBOW_TOO_BENT = 150   # elbow angle below this = too bent
ARM_TOO_LOW = 60    # shoulder angle below this at top = not raised enough
ARM_TOO_HIGH = 100   # shoulder angle above this = raised too high
SYMMETRY_MAX = 20    # angle difference between arms above this = uneven

# ── Inference settings ────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.80   # below this = model says "uncertain"
MIN_VISIBILITY = 0.65   # skip landmarks below this confidence

# ── Logging ───────────────────────────────────────────────────────────
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

log_filename = datetime.now().strftime("gymform_%Y%m%d_%H%M%S.log")
LOG_PATH = os.path.join(LOG_DIR, log_filename)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler()
    ]
)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
