import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os

def preprocess(df):
    df = df.dropna()
    
    # Separate target
    target = df["Churn"]
    df = df.drop("Churn", axis=1)

    categorical = df.select_dtypes(include=["object"]).columns
    numerical = df.select_dtypes(include=["int64", "float64"]).columns

    # One-hot encoding
    df = pd.get_dummies(df, columns=categorical)

    # Scaling
    scaler = StandardScaler()
    df[numerical] = scaler.fit_transform(df[numerical])

    # Ensure models folder exists
    os.makedirs("models", exist_ok=True)

    # ✅ Save scaler
    joblib.dump(scaler, "models/scaler.pkl")

    # Save numerical columns
    joblib.dump(numerical.tolist(), "models/numerical_cols.pkl")

    # Add target back
    df["Churn"] = target.map({"Yes": 1, "No": 0})

    return df


if __name__ == "__main__":
    df = pd.read_csv("data/raw/TelcoCustomer_Churn.csv")
    df = preprocess(df)

    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/train.csv", index=False)