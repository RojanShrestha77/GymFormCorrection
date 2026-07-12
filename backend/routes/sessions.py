from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from typing import List
from backend.database import get_db
from backend.dependencies.auth import get_current_user
from backend.schema.session import SaveSessionRequest, SaveSessionResponse, SessionResponse, UpdateNotesRequest
from backend.services.session_service import SessionService
from backend.repositories.session_repository import SessionRepository
from backend.repositories.user_repository import UserRepository

router = APIRouter()


@router.post("/sessions", response_model=SaveSessionResponse)
def save_session(
    request: SaveSessionRequest,
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    try:
        session_service = SessionService(SessionRepository(db), UserRepository(db))
        session_id = session_service.save_session(
            user_email=current_user,
            exercise=request.exercise,
            start_time=request.start_time,
            end_time=request.end_time,
            total_reps=request.total_reps,
            correct_reps=request.correct_reps,
            incorrect_reps=request.incorrect_reps,
        )
        return SaveSessionResponse(session_id=session_id, message="Session saved successfully")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/sessions", response_model=List[SessionResponse])
def get_sessions(
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    try:
        session_service = SessionService(SessionRepository(db), UserRepository(db))
        return session_service.get_user_sessions(current_user)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error")


@router.patch("/sessions/{session_id}/notes")
def update_notes(
    session_id: int,
    body: UpdateNotesRequest,
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    try:
        SessionService(SessionRepository(db), UserRepository(db)).update_notes(
            current_user, session_id, body.notes
        )
        return {"message": "Notes saved"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.delete("/sessions/{session_id}")
def delete_session(
    session_id: int,
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    try:
        session_service = SessionService(SessionRepository(db), UserRepository(db))
        session_service.delete_session(current_user, session_id)
        return {"message": "Session deleted"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error")
