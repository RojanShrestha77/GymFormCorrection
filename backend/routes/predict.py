import config as gymconfig
from features import get_feedback
from fastapi import APIRouter, HTTPException
from schema.predict import PredictRequest, PredictResponse
from ml.loader import get_model

router = APIRouter()  # creates a router object to register endpoints

# recevies 135 features from frontend
# runs ML prediction using you trained model
# calu


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
        # this gives the final answer "correct_form" : "incorrect_form"
        prediction = model.predict([request.features])[0]
        # this give the probabilites for each class
        probabilities = model.predict_proba([request.features])[0]
        # takes the highest porbability convert to percentage and round it
        confidence = round(max(probabilities) * 100, 1)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    return PredictResponse(
        prediction=prediction,
        confidence=confidence,
        feedback=[]
    )
