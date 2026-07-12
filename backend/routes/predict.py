from fastapi import APIRouter, HTTPException, Depends
from backend.schema.predict import PredictRequest, PredictResponse
from backend.services.prediction_service import PredictionService
from backend.dependencies.auth import get_current_user

router = APIRouter()


@router.post("/predict", response_model=PredictResponse)
def predict(
    request: PredictRequest,
    current_user: str = Depends(get_current_user),
):
    try:
        prediction, confidence, feedback = PredictionService.predict_exercise(
            request.features
        )
        print(f"[PREDICT] label={prediction} confidence={confidence:.1f}% feedback={feedback}")
        return PredictResponse(
            prediction=prediction,
            confidence=confidence,
            feedback=feedback,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error")
