from pydantic import BaseModel
from typing import List


class PredictRequest(BaseModel):
    features: List[float]
    exercise: str = "lateral_raise"


class PredictResponse(BaseModel):
    prediction: str        # "correct" or "incorrect"
    confidence: float      # 0.0 to 100.0
    feedback: List[str]    # ["Raise arms higher", ...]
