# connection to ur databse(used to run queries)
from sqlalchemy.orm import Session
# user -> your database table/model (probably email, password_hash etc)
from models import User
from typing import Optional           # might return something or none


class UserRepository:
    def __init__(self, db: Session):
        self.db = db    # giving access to the database lets you read insert update delete

    def find_by_email(self, email: str) -> Optional[User]:
        return self.db.query(User).filter(User.email == email).first()

    def create_user(self, email: str, password_hash: str) -> User:
        user = User(email=email, password_hash=password_hash)
        self.db.add(user)
        self.db.commit()
        self.db.refresh(user)
        return user
