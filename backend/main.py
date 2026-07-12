from backend.backend_config import ALLOWED_ORIGINS, PROJECT_NAME, VERSION
from backend.routes.auth import router as auth_router
from backend.routes.predict import router as predict_router
from backend.routes.sessions import router as sessions_router
from backend.ml.loader import load_model
from backend.models import Base
from backend.database import engine
from backend.rate_limiter import limiter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
import sys
import os

# MUST be first - adds parent dir to path before any local imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Ensure uploads directory exists
UPLOADS_DIR = os.path.join(parent_dir, "uploads", "profiles")
os.makedirs(UPLOADS_DIR, exist_ok=True)

# Auto-create all tables on startup
Base.metadata.create_all(bind=engine)

# Lightweight column migration — adds new columns to existing tables without Alembic
def _run_migrations():
    from sqlalchemy import text, inspect
    with engine.connect() as conn:
        inspector = inspect(engine)
        existing = {col["name"] for col in inspector.get_columns("users")}
        if "profile_image" not in existing:
            conn.execute(text("ALTER TABLE users ADD COLUMN profile_image VARCHAR"))
            conn.commit()

    with engine.connect() as conn:
        inspector = inspect(engine)
        existing_s = {col["name"] for col in inspector.get_columns("sessions")}
        if "notes" not in existing_s:
            conn.execute(text("ALTER TABLE sessions ADD COLUMN notes VARCHAR"))
            conn.commit()

_run_migrations()

app = FastAPI(title=PROJECT_NAME, version=VERSION)

# Rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    load_model()


app.include_router(predict_router, prefix="/api/v1")
app.include_router(auth_router, prefix="/api/v1/auth")
app.include_router(sessions_router, prefix="/api/v1")

# Serve uploaded profile images as static files
app.mount("/uploads", StaticFiles(directory=os.path.join(parent_dir, "uploads")), name="uploads")


@app.get("/health")
def health():
    return {"status": "ok"}
