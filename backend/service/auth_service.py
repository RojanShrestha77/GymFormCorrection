# used to hash and verify password securely
from passlib.context import CryptContext
from jose import jwt  # used to create jwt tokens
from datetime import datetime, timedelta  # used to set token expiration time
from backend_config import SECRET_KEY, ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES
# SECRET_KEY → used to sign tokens (VERY important),ALGORITHM → usually "HS256",ACCESS_TOKEN_EXPIRE_MINUTES → how long login stays valid
from repositories.user_repository import UserRepository


class AuthService:
    def __init__(self, user_repo: UserRepository):
        self.user_repo = user_repo
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

    def register_user(self, email: str, password: str) -> str:
        # check if user exists
        if self.user_repo.find_by_email(email):
            raise ValueError("Email already registered")

        # Hash password and create user
        password_hash = self.pwd_context.hash(password)
        user = self.user_repo.create_user(email, password_hash)

        # Return token
        return self._create_token(email)

    def login_user(self, email: str, password: str) -> str:
        user = self.user_repo.find_by_email(email)
        if not user or not self.pwd_context.verify(password, user.password_hash):
            raise ValueError("Invalid email or password")

        return self._create_token(email)

    def _create_token(self, email: str) -> str:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        return jwt.encode(
            {"sub": email, "exp": expire},
            SECRET_KEY,
            algorithm=ALGORITHM
        )
