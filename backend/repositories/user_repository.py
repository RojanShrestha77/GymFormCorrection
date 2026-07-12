from sqlalchemy.orm import Session
from backend.models import User
from typing import Optional


class UserRepository:
    def __init__(self, db: Session):
        self.db = db

    def find_by_email(self, email: str) -> Optional[User]:
        return self.db.query(User).filter(User.email == email).first()

    def create_user(self, email: str, password_hash: str) -> User:
        user = User(email=email, password_hash=password_hash)
        self.db.add(user)
        self.db.commit()
        self.db.refresh(user)
        return user

    def find_by_id(self, user_id: int) -> Optional[User]:
        return self.db.query(User).filter(User.id == user_id).first()

    def update_password(self, email: str, new_hash: str) -> None:
        user = self.find_by_email(email)
        if user:
            user.password_hash = new_hash
            self.db.commit()

    def delete_by_email(self, email: str) -> None:
        user = self.find_by_email(email)
        if user:
            self.db.delete(user)
            self.db.commit()
