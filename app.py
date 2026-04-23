import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier

# -------------------------------
# Load Dataset
# -------------------------------
df = pd.read_csv("cleaned_loan_default.csv")

# -------------------------------
# Clean column names
# -------------------------------
df.columns = df.columns.str.strip()

# -------------------------------
# Select Features (based on your dataset)
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

target = 'Status'

# -------------------------------
# Convert target to numeric
# -------------------------------
df[target] = df[target].astype('category').cat.codes

# -------------------------------
# Convert features to numeric
# -------------------------------
df[features] = df[features].apply(pd.to_numeric, errors='coerce')

# -------------------------------
# Fill missing values (IMPORTANT)
# -------------------------------
for col in features:
    if df[col].isnull().all():
        df[col] = df[col].fillna(0)
    else:
        df[col] = df[col].fillna(df[col].median())

# -------------------------------
# Final safety check
# -------------------------------
if df[features].isnull().sum().sum() > 0:
    st.error("Data still contains NaN values!")
    st.stop()

X = df[features]
y = df[target]

# -------------------------------
# Train Model (NaN-safe model)
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = HistGradientBoostingClassifier()
model.fit(X_train, y_train)

# -------------------------------
# Streamlit UI
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

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High Risk (Fraud/Default)\n\nProbability: {probability:.2f}")
    else:
        st.success(f"✅ Low Risk\n\nProbability: {probability:.2f}")