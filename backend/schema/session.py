from pydantic import BaseModel
from datetime import datetime
from typing import Optional


class SaveSessionRequest(BaseModel):
    exercise: str
    start_time: str  # ISO format
    end_time: str
    total_reps: int
    correct_reps: int
    incorrect_reps: int


class SaveSessionResponse(BaseModel):
    session_id: int
    message: str


class UpdateNotesRequest(BaseModel):
    notes: str


class SessionResponse(BaseModel):
    id: int
    exercise: str
    start_time: datetime
    end_time: Optional[datetime]
    total_reps: int
    correct_reps: int
    form_score: Optional[float]
    notes: Optional[str] = None

    class Config:
        from_attributes = True
