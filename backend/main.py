from backend_config import ALLOWED_ORIGINS, PROJECT_NAME, VERSION
from routes.auth import router as auth_router
from routes.predict import router as predict_router
from ml.loader import load_model
from models import Base
from database import engine
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
import sys
import os

# MUST be first - adds parent dir to path before any local imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


# Auto-create all tables on startup
Base.metadata.create_all(bind=engine)

app = FastAPI(title=PROJECT_NAME, version=VERSION)

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


@app.get("/health")
def health():
    return {"status": "ok"}
