import os
from dotenv import load_dotenv

load_dotenv()

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
TFLITE_MODEL_PATH = os.path.join(PROJECT_ROOT, "lateral_raise_model.tflite")

# API Settings
API_V1_PREFIX = "/api/v1"
PROJECT_NAME = "GymForm AI API"
VERSION = "1.0.0"
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

# CORS — comma-separated in .env
_raw_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8081")
ALLOWED_ORIGINS = [o.strip() for o in _raw_origins.split(",") if o.strip()]

# Auth — fail fast if SECRET_KEY is missing
SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    raise RuntimeError(
        "SECRET_KEY is not set. Add it to your .env file.\n"
        "Generate one with: openssl rand -hex 32"
    )
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "10080"))

# Database
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./gymform.db")
