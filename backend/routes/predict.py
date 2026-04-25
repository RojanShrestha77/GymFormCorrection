import config as gymconfig
from features import get_feedback, get_joint_angle
from fastapi import APIRouter, HTTPException
from schema.predict import PredictRequest, PredictResponse
from ml.loader import get_model

router = APIRouter()  # creates a router object to register endpoints


class MockLandmark:
    """Mock landmark object to reconstruct from features for feedback generation"""
    def __init__(self, x, y, z, visibility):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility


def reconstruct_landmarks_from_features(features):
    """
    Reconstruct landmark objects from the 135 features.
    Features format: [avg_shoulder_angle, avg_elbow_angle, symmetry, x_0, y_0, z_0, v_0, x_1, y_1, z_1, v_1, ...]
    """
    # Skip first 3 angle features, then extract raw landmarks (33 landmarks × 4 values each = 132)
    raw_data = features[3:]
    landmarks = []
    
    for i in range(0, len(raw_data), 4):
        landmarks.append(MockLandmark(
            x=raw_data[i],
            y=raw_data[i+1],
            z=raw_data[i+2],
            visibility=raw_data[i+3]
        ))
    
    return landmarks


@router.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    bundle = get_model()  # calls the get_model from the loader
    if bundle is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if len(request.features) != 135:
        raise HTTPException(
            status_code=400,
            detail=f"Expected 135 features, got {len(request.features)}"
        )

    model = bundle if not isinstance(bundle, dict) else bundle["model"]

    try:
        # this gives the final answer "correct" or "incorrect"
        prediction = model.predict([request.features])[0]
        # this gives the probabilities for each class
        probabilities = model.predict_proba([request.features])[0]
        # takes the highest probability convert to percentage and round it
        confidence = round(max(probabilities) * 100, 1)
        
        # Generate feedback from the features
        landmarks = reconstruct_landmarks_from_features(request.features)
        angles = get_joint_angle(landmarks)
        feedback_messages = get_feedback(angles, gymconfig)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    return PredictResponse(
        prediction=prediction,
        confidence=confidence,
        feedback=feedback_messages
    )
