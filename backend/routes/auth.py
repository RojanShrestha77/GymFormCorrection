from fastapi import APIRouter, HTTPException, Depends, Request, UploadFile, File
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session
from typing import Optional
import os, uuid
from backend.database import get_db
from backend.services.auth_service import AuthService
from backend.repositories.user_repository import UserRepository
from backend.repositories.session_repository import SessionRepository
from backend.dependencies.auth import get_current_user
from backend.rate_limiter import limiter
from backend.backend_config import PROJECT_ROOT

UPLOADS_DIR = os.path.join(PROJECT_ROOT, "uploads", "profiles")
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_IMAGE_BYTES = 5 * 1024 * 1024  # 5 MB

router = APIRouter()


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class RefreshRequest(BaseModel):
    refresh_token: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class ForgotPasswordRequest(BaseModel):
    email: EmailStr


class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str


class MeResponse(BaseModel):
    email: str
    created_at: Optional[str] = None
    total_sessions: int = 0
    total_reps: int = 0
    avg_accuracy: float = 0.0
    profile_image_url: Optional[str] = None


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str


class DeleteAccountRequest(BaseModel):
    password: str


@router.post("/register", response_model=TokenResponse)
@limiter.limit("5/minute")
def register(request: Request, body: RegisterRequest, db: Session = Depends(get_db)):
    try:
        tokens = AuthService(UserRepository(db)).register_user(body.email, body.password)
        return TokenResponse(**tokens)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/login", response_model=TokenResponse)
@limiter.limit("10/minute")
def login(request: Request, body: LoginRequest, db: Session = Depends(get_db)):
    try:
        tokens = AuthService(UserRepository(db)).login_user(body.email, body.password)
        return TokenResponse(**tokens)
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/refresh", response_model=TokenResponse)
def refresh(body: RefreshRequest, db: Session = Depends(get_db)):
    """Exchange a valid refresh token for a new access + refresh token pair (rotation)."""
    try:
        tokens = AuthService(UserRepository(db)).refresh_access_token(body.refresh_token)
        return TokenResponse(**tokens)
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/me", response_model=MeResponse)
def me(current_user: str = Depends(get_current_user), db: Session = Depends(get_db)):
    """Returns profile info used on startup and the profile page."""
    user = UserRepository(db).find_by_email(current_user)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    sessions = SessionRepository(db).find_by_user_id(user.id)
    total_sessions = len(sessions)
    total_reps = sum(s.total_reps for s in sessions)
    scored = [s for s in sessions if s.form_score is not None]
    avg_accuracy = sum(s.form_score for s in scored) / len(scored) if scored else 0.0

    image_url = f"/uploads/profiles/{user.profile_image}" if user.profile_image else None
    return MeResponse(
        email=user.email,
        created_at=user.created_at.isoformat() if user.created_at else None,
        total_sessions=total_sessions,
        total_reps=total_reps,
        avg_accuracy=round(avg_accuracy, 1),
        profile_image_url=image_url,
    )


@router.post("/change-password")
def change_password(
    body: ChangePasswordRequest,
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Verifies current password then updates to new password."""
    try:
        AuthService(UserRepository(db)).change_password(
            email=current_user,
            current_password=body.current_password,
            new_password=body.new_password,
        )
        return {"message": "Password changed successfully."}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/account")
def delete_account(
    body: DeleteAccountRequest,
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Permanently deletes the authenticated user and all their data."""
    try:
        AuthService(UserRepository(db)).delete_account(current_user, body.password)
        return {"message": "Account deleted successfully."}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/profile-image")
async def upload_profile_image(
    file: UploadFile = File(...),
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Uploads and saves a profile picture for the authenticated user."""
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(status_code=400, detail="Only JPEG, PNG and WebP images are allowed.")

    contents = await file.read()
    if len(contents) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=400, detail="Image must be under 5 MB.")

    ext = file.filename.rsplit(".", 1)[-1].lower() if file.filename and "." in file.filename else "jpg"
    filename = f"{uuid.uuid4().hex}.{ext}"

    # Delete old profile image if it exists
    user = UserRepository(db).find_by_email(current_user)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if user.profile_image:
        old_path = os.path.join(UPLOADS_DIR, user.profile_image)
        if os.path.exists(old_path):
            os.remove(old_path)

    # Save new file
    os.makedirs(UPLOADS_DIR, exist_ok=True)
    save_path = os.path.join(UPLOADS_DIR, filename)
    with open(save_path, "wb") as f:
        f.write(contents)

    # Persist filename in DB
    user.profile_image = filename
    db.commit()

    return {"profile_image_url": f"/uploads/profiles/{filename}"}


@router.post("/forgot-password")
@limiter.limit("3/minute")
def forgot_password(request: Request, body: ForgotPasswordRequest, db: Session = Depends(get_db)):
    """Sends a password reset email. Always returns 200 to prevent email enumeration."""
    try:
        AuthService(UserRepository(db)).send_password_reset(body.email)
    except Exception:
        pass  # Never reveal whether the email exists or what went wrong
    return {"message": "If that email exists, a reset link has been sent."}


@router.post("/reset-password")
def reset_password(body: ResetPasswordRequest, db: Session = Depends(get_db)):
    """Verifies reset token and updates password."""
    try:
        AuthService(UserRepository(db)).reset_password(body.token, body.new_password)
        return {"message": "Password updated successfully."}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error")
