import streamlit as st
import requests

API_URL = "https://churn-mlops-qqku.onrender.com/predict"

st.set_page_config(page_title="Churn Predictor", layout="centered")

st.title("📊 Customer Churn Prediction")

st.write("Enter customer details to predict churn probability")

# --- INPUT FORM ---
gender = st.selectbox("Gender", ["Male", "Female"])
senior = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Partner", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])

tenure = st.slider("Tenure (months)", 0, 72, 12)

phone = st.selectbox("Phone Service", ["Yes", "No"])
multiple = st.selectbox("Multiple Lines", ["Yes", "No"])

internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

online_sec = st.selectbox("Online Security", ["Yes", "No"])
backup = st.selectbox("Online Backup", ["Yes", "No"])
device = st.selectbox("Device Protection", ["Yes", "No"])
tech = st.selectbox("Tech Support", ["Yes", "No"])

tv = st.selectbox("Streaming TV", ["Yes", "No"])
movies = st.selectbox("Streaming Movies", ["Yes", "No"])

contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
payment = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
)

monthly = st.number_input("Monthly Charges", value=70.0)
total = st.number_input("Total Charges", value=800.0)

# --- PREDICT BUTTON ---
if st.button("Predict Churn"):
    payload = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple,
        "InternetService": internet,
        "OnlineSecurity": online_sec,
        "OnlineBackup": backup,
        "DeviceProtection": device,
        "TechSupport": tech,
        "StreamingTV": tv,
        "StreamingMovies": movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
    }

    try:
        response = requests.post(API_URL, json=payload)
        result = response.json()

        prob = result["churn_probability"]

        st.subheader("Prediction Result")

        if prob > 0.5:
            st.error(f"⚠️ High Churn Risk: {prob:.2f}")
        else:
            st.success(f"✅ Low Churn Risk: {prob:.2f}")

    except Exception as e:
        st.error(f"Error: {e}")