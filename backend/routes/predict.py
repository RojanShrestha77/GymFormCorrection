import config as gymconfig
from features import get_feedback
from fastapi import APIRouter, HTTPException
from schemas.predict import PredictRequest, PredictResponse
from ml.loader import get_model
import sys
import os

sys.path.append(r"C:\Users\LOQ\Desktop\GymForm")

router = APIRouter()


@router.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    bundle = get_model()
    if bundle is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if len(request.features) != 135:
        raise HTTPException(
            status_code=400,
            detail=f"Expected 135 features, got {len(request.features)}"
        )

    model = bundle if not isinstance(bundle, dict) else bundle["model"]

    try:
        prediction = model.predict([request.features])[0]
        probabilities = model.predict_proba([request.features])[0]
        confidence = round(max(probabilities) * 100, 1)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    return PredictResponse(
        prediction=prediction,
        confidence=confidence,
        feedback=[]
    )
