import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

from config import PROCESSED_DATA_DIR, MODELS_DIR

data = pd.read_csv(PROCESSED_DATA_DIR / "processed_data.csv")

features = data.drop(columns=["GPA"])

scaler = StandardScaler()
scaler.fit(features)

scaler_path = MODELS_DIR / "scaler.pkl"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(scaler, scaler_path)

print(f"The Scaler is saved to: {scaler_path}")
