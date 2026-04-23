import streamlit as st
import pandas as pd
import joblib
import os

# -------------------------------
# Debug: show files
# -------------------------------
st.write("Files in directory:", os.listdir())

# -------------------------------
# Check files exist
# -------------------------------
if not os.path.exists("model.pkl"):
    st.error("model.pkl not found")
    st.stop()

if not os.path.exists("scaler.pkl"):
    st.error("scaler.pkl not found")
    st.stop()

# -------------------------------
# Load model & scaler
# -------------------------------
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# -------------------------------
# Feature list
# -------------------------------
features = [
    'loan_amount',
    'income',
    'Credit_Score',
    'LTV',
    'dtir1',
    'property_value',
    'term',
    'open_credit',
    'age'
]

# -------------------------------
# UI
# -------------------------------
st.title("💰 Loan Fraud / Default Detection App")

st.write("Enter applicant details:")

loan_amount = st.number_input("Loan Amount")
income = st.number_input("Income")
credit_score = st.number_input("Credit Score")
ltv = st.number_input("LTV")
dtir1 = st.number_input("DTI Ratio")
property_value = st.number_input("Property Value")
term = st.number_input("Loan Term")
open_credit = st.number_input("Open Credit")
age = st.number_input("Age")

# -------------------------------
# Prediction
# -------------------------------
if st.button("Check Risk"):

    input_data = pd.DataFrame([[ 
        loan_amount, income, credit_score, ltv,
        dtir1, property_value, term, open_credit, age
    ]], columns=features)

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High Risk (Fraud/Default)\n\nProbability: {probability:.2f}")
    else:
        st.success(f"✅ Low Risk\n\nProbability: {probability:.2f}")
