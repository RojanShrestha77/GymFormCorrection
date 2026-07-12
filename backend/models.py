from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from backend.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    profile_image = Column(String, nullable=True)

    # Relationship - one user has many sessions
    sessions = relationship("WorkoutSession", back_populates="user", cascade="all, delete-orphan")


class WorkoutSession(Base):
    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    exercise = Column(String, default="lateral_raise")
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime, nullable=True)
    total_reps = Column(Integer, default=0)
    correct_reps = Column(Integer, default=0)
    form_score = Column(Float, nullable=True)

    notes = Column(String, nullable=True)

    # Relationship - session belongs to a user
    user = relationship("User", back_populates="sessions")
