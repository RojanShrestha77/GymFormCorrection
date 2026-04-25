import os

BASE_DIR = r"C:\Users\LOQ\Desktop\GymForm"
MODEL_PATH = os.path.join(BASE_DIR, "lateral_raise-model.pkl")


SECRET_KEY = "change-this-to-a-random-secret-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24

DATABASE_URL = "sqllite:///./gymform.db"
