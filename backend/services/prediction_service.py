import numpy as np
from typing import Tuple, List

from backend.ml.loader import get_model
import config as gymconfig


class PredictionService:
    @staticmethod
    def predict_exercise(features: List[float]) -> Tuple[str, float, List[str]]:
        """
        Predict exercise form from a flattened TCN feature vector.

        Expects 180 floats: 30 frames × 6 biomechanical angles, in row-major order
        (frame_0_feat_0, ..., frame_0_feat_5, frame_1_feat_0, ..., frame_29_feat_5).
        """
        expected = gymconfig.SEQ_LEN * gymconfig.N_FEATURES  # 30 × 6 = 180
        if len(features) != expected:
            raise ValueError(
                f"Expected {expected} features (30 frames × 6), got {len(features)}"
            )

        interp = get_model()
        if interp is None:
            raise RuntimeError(
                "TFLite model not loaded. Run train_model.py to generate the model, "
                "then restart the server."
            )

        # Reshape flat list → (1, 30, 6)
        X = np.array(features, dtype=np.float32).reshape(
            1, gymconfig.SEQ_LEN, gymconfig.N_FEATURES
        )

        input_details = interp.get_input_details()
        output_details = interp.get_output_details()

        interp.set_tensor(input_details[0]["index"], X)
        interp.invoke()

        # TFLite reorders outputs alphabetically: "error" [1,4] before "form" [1,1].
        # Detect by shape so this is robust to future reorders.
        raw = [interp.get_tensor(o["index"]) for o in output_details]
        form_tensor  = next(t for t in raw if t.shape[-1] == 1)
        error_tensor = next(t for t in raw if t.shape[-1] == gymconfig.N_CLASSES)

        form_conf  = float(form_tensor[0][0])
        error_probs = error_tensor[0]
        pred_idx   = int(np.argmax(error_probs))
        pred_label = gymconfig.LABELS[pred_idx]
        confidence = round(float(error_probs[pred_idx]) * 100, 1)

        feedback = PredictionService._feedback_for(pred_label, form_conf)

        return pred_label, confidence, feedback

    @staticmethod
    def _feedback_for(label: str, form_conf: float) -> List[str]:
        if form_conf >= 0.5:
            return ["Good form! Keep it up."]
        messages = {
            "elbow_bent":      ["Keep your elbows straighter — aim for ~160°"],
            "not_high_enough": ["Raise your arms to shoulder height"],
            "torso_lean":      ["Keep your torso upright, don't lean sideways"],
        }
        return messages.get(label, ["Check your form"])
