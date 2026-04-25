import sys
import os

# Add parent directory to path so we can import config.py and features.py from GymForm/
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from fastapi import FastAPI
from ml.loader import load_model
from routes.predict import router as predict_router
from routes.auth import router as auth_router

app = FastAPI(title="GymForm API", version="1.0.0")


@app.on_event("startup")
def startup():
    load_model()


app.include_router(predict_router, prefix="/api/v1")
app.include_router(auth_router, prefix="/api/v1/auth")


@app.get("/health")
def health():
    return {"status": "ok"}
