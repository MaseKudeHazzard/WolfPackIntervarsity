import streamlit as st
import requests
import plotly.express as px
import pandas as pd
import sqlite3
import shap
import joblib
import numpy as np

st.title("Micro-Loan Underwriting Assistant Dashboard")

# API base URL
BASE_URL = "http://127.0.0.1:8000"

# Load SHAP explainer for visualizations
model = joblib.load("logistic_regression_model.pkl")
scaler = joblib.load("scaler.pkl")
background_data = joblib.load("background_data.pkl")
explainer = shap.LinearExplainer(model, background_data)

# Sidebar for navigation
page = st.sidebar.selectbox("Select Page", ["Loan Application", "User Progress", "Repayment"])

if page == "Loan Application":
    st.header("Apply for a Loan")
    with st.form("loan_application"):
        user_id = st.text_input("User ID", "user123")
        loan_amount = st.number_input("Loan Amount", min_value=0.01, value=1000.0)
        transaction_frequency = st.number_input("Transaction Frequency (per month)", min_value=0.0, value=15.0)
        avg_transaction_amount = st.number_input("Average Transaction Amount ($)", min_value=0.0, value=100.0)
        utility_payment_consistency = st.slider("Utility Payment Consistency (0-1)", 0.0, 1.0, 0.9)
        airtime_topup_frequency = st.number_input("Airtime Top-up Frequency (per month)", min_value=0.0, value=8.0)
        submit = st.form_submit_button("Apply")

        if submit:
            payload = {
                "user_id": user_id,
                "loan_amount": loan_amount,
                "transaction_frequency": transaction_frequency,
                "avg_transaction_amount": avg_transaction_amount,
                "utility_payment_consistency": utility_payment_consistency,
                "airtime_topup_frequency": airtime_topup_frequency
            }
            response = requests.post(f"{BASE_URL}/loan/apply", json=payload)
            if response.status_code == 200:
                result = response.json()
                st.success(f"Loan {result['decision']}!")
                st.json(result)
                # Visualize SHAP explanation
                features = [transaction_frequency, avg_transaction_amount, utility_payment_consistency, airtime_topup_frequency]
                features_scaled = scaler.transform([features])
                shap_values = explainer(features_scaled)[0].values
                shap_df = pd.DataFrame({
                    "Feature": ["Transaction Frequency", "Avg Transaction Amount", "Utility Payment Consistency", "Airtime Top-up Frequency"],
                    "SHAP Value": shap_values
                })
                fig = px.bar(shap_df, x="SHAP Value", y="Feature", title="SHAP Feature Importance")
                st.plotly_chart(fig)
            else:
                st.error(f"Error: {response.json()['detail']}")

elif page == "User Progress":
    st.header("User Progress")
    user_id = st.text_input("User ID", "user123")
    if st.button("Get Progress"):
        response = requests.get(f"{BASE_URL}/user/progress/{user_id}")
        if response.status_code == 200:
            result = response.json()
            st.json(result)
            # Visualize alternative data
            data = result["alternative_data"]
            df = pd.DataFrame([data])
            fig = px.bar(df, x=df.columns, y=df.values[0], title="Alternative Data Profile")
            st.plotly_chart(fig)
            # Visualize repayment history
            progress_map = result["gamification"]["progress_map"]
            if progress_map:
                progress_df = pd.DataFrame(progress_map)
                fig = px.scatter(progress_df, x="date", y="amount", color="status", title="Repayment History")
                st.plotly_chart(fig)
            # Display gamification metrics
            st.metric("Repayment Streak", result["gamification"]["repayment_streak"])
            st.metric("Points Earned", result["gamification"]["points_earned"])
            st.write("Badges:", ", ".join(result["gamification"]["badges_earned"]))
        else:
            st.error(f"Error: {response.json()['detail']}")

elif page == "Repayment":
    st.header("Record Repayment")
    with st.form("repayment"):
        user_id = st.text_input("User ID", "user123")
        loan_id = st.text_input("Loan ID", "Luser123_1726279987")
        payment_date = st.date_input("Payment Date")
        amount = st.number_input("Repayment Amount", min_value=0.01, value=1000.0)
        submit = st.form_submit_button("Submit Repayment")

        if submit:
            payload = {
                "user_id": user_id,
                "loan_id": loan_id,
                "payment_date": payment_date.strftime("%Y-%m-%d"),
                "amount": amount
            }
            response = requests.post(f"{BASE_URL}/repayment/record", json=payload)
            if response.status_code == 200:
                result = response.json()
                st.success("Repayment recorded!")
                st.json(result)
            else:
                st.error(f"Error: {response.json()['detail']}")
