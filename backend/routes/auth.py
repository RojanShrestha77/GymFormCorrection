from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, EmailStr
from passlib.context import CryptContext
from jose import jwt
from datetime import datetime, timedelta
from config import SECRET_KEY, ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES

router = APIRouter()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Temporary in-memory user store — replace with DB in Phase 5
fake_users = {}


class RegisterRequest(BaseModel):
    email: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


def create_token(email: str) -> str:
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    return jwt.encode(
        {"sub": email, "exp": expire},
        SECRET_KEY,
        algorithm=ALGORITHM
    )


@router.post("/register", response_model=TokenResponse)
def register(request: RegisterRequest):
    if request.email in fake_users:
        raise HTTPException(status_code=400, detail="Email already registered")
    fake_users[request.email] = pwd_context.hash(request.password)
    return TokenResponse(access_token=create_token(request.email))


@router.post("/login", response_model=TokenResponse)
def login(request: LoginRequest):
    hashed = fake_users.get(request.email)
    if not hashed or not pwd_context.verify(request.password, hashed):
        raise HTTPException(
            status_code=401, detail="Invalid email or password")
    return TokenResponse(access_token=create_token(request.email))
