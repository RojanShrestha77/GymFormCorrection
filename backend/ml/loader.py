import os
import tensorflow as tf

from backend.backend_config import TFLITE_MODEL_PATH

_interpreter = None


def load_model():
    global _interpreter
    if not os.path.exists(TFLITE_MODEL_PATH):
        print(
            f"[loader] TFLite model not found at {TFLITE_MODEL_PATH}. "
            "Run train_model.py first, then restart the server."
        )
        return
    _interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
    _interpreter.allocate_tensors()
    print(f"[loader] TFLite model loaded: {TFLITE_MODEL_PATH}")


def get_model() -> tf.lite.Interpreter | None:
    return _interpreter
