import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
MODEL_PATH = os.path.join(PROJECT_ROOT, "lateral_raise_model.pkl")

# API Settings
API_V1_PREFIX = "/api/v1"
PROJECT_NAME = "GymForm AI API"
VERSION = "1.0.0"

# CORS Settings
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:8081",  # Expo default
    "http://localhost:19000",  # Expo default
    "http://localhost:19006",  # Expo web
]

# Auth Settings
SECRET_KEY = "your-secret-key-change-in-production-use-openssl-rand-hex-32"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 days


# Database Settings
DATABASE_URL = "postgresql://postgres:yourpassword@localhost:5432/gymform_db"
