import streamlit as st
import requests

API_URL = "http://127.0.0.1:5000/predict"

st.set_page_config(page_title="Fraud Detection", layout="wide")

st.title("🔍 Fraud Detection Dashboard")

st.markdown("Enter transaction details to check fraud probability")

# Inputs
col1, col2 = st.columns(2)

with col1:
    step = st.number_input("Step", min_value=1, value=1)
    txn_type = st.selectbox("Transaction Type", ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT"])
    amount = st.number_input("Amount", min_value=0.0, value=1000.0)

with col2:
    old_org = st.number_input("Old Balance Sender", value=5000.0)
    new_org = st.number_input("New Balance Sender", value=4000.0)
    old_dest = st.number_input("Old Balance Receiver", value=0.0)
    new_dest = st.number_input("New Balance Receiver", value=1000.0)

if st.button("🚀 Predict Fraud"):
    raw = {
        "step": step,
        "type": txn_type,
        "amount": amount,
        "oldbalanceOrg": old_org,
        "newbalanceOrig": new_org,
        "oldbalanceDest": old_dest,
        "newbalanceDest": new_dest
    }

    try:
        response = requests.post(API_URL, json=raw)

        st.write("Status Code:", response.status_code)
        st.write("Raw Response:", response.text)

        if response.status_code == 200:
            result = response.json()

            st.success("Prediction Successful")
            st.write(result)

        else:
            st.error(f"API Error: {response.text}")

    except Exception as e:
        st.error(f"API not running: {e}")