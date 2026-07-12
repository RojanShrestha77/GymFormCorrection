from sqlalchemy.orm import Session
from backend.models import WorkoutSession
from typing import List, Optional
from datetime import datetime


class SessionRepository:
    def __init__(self, db: Session):
        self.db = db

    def create_session(
        self,
        user_id: int,
        exercise: str,
        start_time: datetime,
        end_time: datetime,
        total_reps: int,
        correct_reps: int,
        form_score: float
    ) -> WorkoutSession:
        """Create a new workout session"""
        session = WorkoutSession(
            user_id=user_id,
            exercise=exercise,
            start_time=start_time,
            end_time=end_time,
            total_reps=total_reps,
            correct_reps=correct_reps,
            form_score=form_score
        )
        self.db.add(session)
        self.db.commit()
        self.db.refresh(session)
        return session

    def find_by_user_id(self, user_id: int) -> List[WorkoutSession]:
        """Get all sessions for a user"""
        return self.db.query(WorkoutSession)\
            .filter(WorkoutSession.user_id == user_id)\
            .order_by(WorkoutSession.start_time.desc())\
            .all()

    def find_by_id(self, session_id: int) -> Optional[WorkoutSession]:
        return self.db.query(WorkoutSession)\
            .filter(WorkoutSession.id == session_id)\
            .first()

    def update_notes(self, session_id: int, user_id: int, notes: str) -> bool:
        session = self.db.query(WorkoutSession)\
            .filter(WorkoutSession.id == session_id, WorkoutSession.user_id == user_id)\
            .first()
        if not session:
            return False
        session.notes = notes
        self.db.commit()
        return True

    def delete_by_id(self, session_id: int, user_id: int) -> bool:
        session = self.db.query(WorkoutSession)\
            .filter(WorkoutSession.id == session_id, WorkoutSession.user_id == user_id)\
            .first()
        if not session:
            return False
        self.db.delete(session)
        self.db.commit()
        return True
