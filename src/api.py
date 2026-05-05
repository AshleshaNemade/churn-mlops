from fastapi import FastAPI, Body
import joblib
import pandas as pd
from src.utils import transform_input

app = FastAPI()

# Load artifacts
model = joblib.load("models/best_model.pkl")
scaler = joblib.load("models/scaler.pkl")
features = joblib.load("models/features.pkl")
numerical_cols = joblib.load("models/numerical_cols.pkl")

@app.get("/")
def home():
    return {"message": "Churn Prediction API is running"}


@app.post("/predict")
def predict(data: dict = Body(...)):
    try:
        df = transform_input(data, features, scaler, numerical_cols)

        prob = model.predict_proba(df)[0][1]

        return {
            "input_received": data,
            "transformed_shape": df.shape,
            "churn_probability": float(prob)
        }

    except Exception as e:
        return {"error": str(e)}