import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils import resample
import pickle 
import os

CSV_PATH   = r"C:\Users\LOQ\Desktop\GymForm\lateral_raise_data.csv"
MODEL_PATH = r"C:\Users\LOQ\Desktop\GymForm\lateral_raise_model.pkl"

#load the csv
