import pickle

# Import from backend/backend_config.py (local config)
from backend_config import MODEL_PATH

model_bundle = None


def load_model():
    global model_bundle
    with open(MODEL_PATH, "rb") as f:
        model_bundle = pickle.load(f)
    print(f"Model loaded: {MODEL_PATH}")


def get_model():
    return model_bundle
