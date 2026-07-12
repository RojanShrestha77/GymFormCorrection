from backend.repositories.session_repository import SessionRepository
from backend.repositories.user_repository import UserRepository
from datetime import datetime


class SessionService:
    def __init__(
        self,
        session_repo: SessionRepository,
        user_repo: UserRepository
    ):
        self.session_repo = session_repo
        self.user_repo = user_repo

    def save_session(
        self,
        user_email: str,
        exercise: str,
        start_time: str,
        end_time: str,
        total_reps: int,
        correct_reps: int,
        incorrect_reps: int
    ) -> int:
        """Save a completed workout session"""
        # Find user
        user = self.user_repo.find_by_email(user_email)
        if not user:
            raise ValueError("User not found")

        # Calculate form score
        form_score = (correct_reps / total_reps * 100) if total_reps > 0 else 0

        # Parse datetime strings
        start_dt = datetime.fromisoformat(start_time)
        end_dt = datetime.fromisoformat(end_time)

        # Create session
        session = self.session_repo.create_session(
            user_id=user.id,
            exercise=exercise,
            start_time=start_dt,
            end_time=end_dt,
            total_reps=total_reps,
            correct_reps=correct_reps,
            form_score=form_score
        )

        return session.id

    def get_user_sessions(self, user_email: str):
        user = self.user_repo.find_by_email(user_email)
        if not user:
            raise ValueError("User not found")
        return self.session_repo.find_by_user_id(user.id)

    def update_notes(self, user_email: str, session_id: int, notes: str) -> None:
        user = self.user_repo.find_by_email(user_email)
        if not user:
            raise ValueError("User not found")
        if not self.session_repo.update_notes(session_id, user.id, notes):
            raise ValueError("Session not found or access denied")

    def delete_session(self, user_email: str, session_id: int) -> None:
        user = self.user_repo.find_by_email(user_email)
        if not user:
            raise ValueError("User not found")
        deleted = self.session_repo.delete_by_id(session_id, user.id)
        if not deleted:
            raise ValueError("Session not found or access denied")
