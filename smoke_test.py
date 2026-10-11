"""Run: python smoke_test.py (checks real model inference without Streamlit)."""
from pathlib import Path
import joblib
import pandas as pd
ROOT = Path(__file__).resolve().parent
model = joblib.load(ROOT / "model.joblib")
data = pd.read_csv(ROOT / "reference_data.csv")
features = list(model.feature_names_in_)
assert all(f in data.columns for f in features), "Missing model features"
pred = model.predict(data.loc[:2, features])
assert len(pred) == 3
print("PASS: trained model loaded and predicted", [round(float(v), 5) for v in pred])
