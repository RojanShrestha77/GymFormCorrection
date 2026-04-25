import os

BASE_DIR = r"C:\Users\LOQ\Desktop\GymForm"  # root folder of the project
MODEL_PATH = os.path.join(BASE_DIR, "lateral_raise_model.pkl")


SECRET_KEY = "change-this-to-a-random-secret-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24

DATABASE_URL = "sqlite:///./gymform.db"
