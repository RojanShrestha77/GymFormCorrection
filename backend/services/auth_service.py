from passlib.context import CryptContext
from jose import jwt, JWTError
from datetime import datetime, timedelta
from backend.backend_config import SECRET_KEY, ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES
from backend.repositories.user_repository import UserRepository
from backend.services.email_service import send_reset_email

_REFRESH_TOKEN_EXPIRE_DAYS = 30
_RESET_TOKEN_EXPIRE_MINUTES = 60  # 1 hour


class AuthService:
    def __init__(self, user_repo: UserRepository):
        self.user_repo = user_repo
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

    def register_user(self, email: str, password: str) -> dict:
        existing = self.user_repo.find_by_email(email)
        if existing:
            raise ValueError("Email already registered")
        password_hash = self.pwd_context.hash(password)
        self.user_repo.create_user(email, password_hash)
        return self._create_token_pair(email)

    def login_user(self, email: str, password: str) -> dict:
        user = self.user_repo.find_by_email(email)
        if not user or not self.pwd_context.verify(password, user.password_hash):
            raise ValueError("Invalid email or password")
        return self._create_token_pair(email)

    def refresh_access_token(self, refresh_token: str) -> dict:
        """Verifies a refresh token and issues a new access + refresh token pair (rotation)."""
        try:
            payload = jwt.decode(refresh_token, SECRET_KEY, algorithms=[ALGORITHM])
            if payload.get("type") != "refresh":
                raise ValueError("Invalid token type")
            email: str = payload.get("sub")
            if not email:
                raise ValueError("Invalid token")
            user = self.user_repo.find_by_email(email)
            if not user:
                raise ValueError("User not found")
            return self._create_token_pair(email)
        except JWTError:
            raise ValueError("Invalid or expired refresh token")

    def _create_token_pair(self, email: str) -> dict:
        return {
            "access_token": self._create_access_token(email),
            "refresh_token": self._create_refresh_token(email),
        }

    def _create_access_token(self, email: str) -> str:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        return jwt.encode(
            {"sub": email, "exp": expire, "type": "access"},
            SECRET_KEY,
            algorithm=ALGORITHM,
        )

    def _create_refresh_token(self, email: str) -> str:
        expire = datetime.utcnow() + timedelta(days=_REFRESH_TOKEN_EXPIRE_DAYS)
        return jwt.encode(
            {"sub": email, "exp": expire, "type": "refresh"},
            SECRET_KEY,
            algorithm=ALGORITHM,
        )

    def change_password(self, email: str, current_password: str, new_password: str) -> None:
        """Verifies current password then updates to new one."""
        user = self.user_repo.find_by_email(email)
        if not user:
            raise ValueError("User not found")
        if not self.pwd_context.verify(current_password, user.password_hash):
            raise ValueError("Current password is incorrect")
        if len(new_password) < 6:
            raise ValueError("New password must be at least 6 characters")
        self.user_repo.update_password(email, self.pwd_context.hash(new_password))

    def send_password_reset(self, email: str) -> None:
        """Generates a 1-hour reset token and emails it. Silent if user not found (security)."""
        user = self.user_repo.find_by_email(email)
        if not user:
            return  # Don't reveal whether email exists
        expire = datetime.utcnow() + timedelta(minutes=_RESET_TOKEN_EXPIRE_MINUTES)
        token = jwt.encode(
            {"sub": email, "exp": expire, "type": "password_reset"},
            SECRET_KEY,
            algorithm=ALGORITHM,
        )
        send_reset_email(email, token)

    def delete_account(self, email: str, password: str) -> None:
        """Verifies password then permanently deletes the user and all their data."""
        user = self.user_repo.find_by_email(email)
        if not user or not self.pwd_context.verify(password, user.password_hash):
            raise ValueError("Incorrect password")
        self.user_repo.delete_by_email(email)

    def reset_password(self, token: str, new_password: str) -> None:
        """Verifies the reset token and updates the user's password."""
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            if payload.get("type") != "password_reset":
                raise ValueError("Invalid token type")
            email: str = payload.get("sub")
            if not email:
                raise ValueError("Invalid token")
        except JWTError:
            raise ValueError("Invalid or expired reset token")

        user = self.user_repo.find_by_email(email)
        if not user:
            raise ValueError("User not found")

        new_hash = self.pwd_context.hash(new_password)
        self.user_repo.update_password(email, new_hash)
