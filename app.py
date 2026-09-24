import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="FraudShield",
    page_icon="🛡️",
    layout="wide"
)

# Load model
model = joblib.load("models/isolation_forest.pkl")
scaler = joblib.load("models/scaler.pkl")

# Load dataset for dashboard statistics
df = pd.read_csv("data/creditcard.csv")

# -----------------------------
# HEADER
# -----------------------------

st.title("🛡️ FraudShield")
st.subheader("Real-Time Transaction Anomaly Detection")

st.write(
    "Machine learning based system for identifying potentially "
    "suspicious financial transactions."
)

st.divider()

# -----------------------------
# DASHBOARD METRICS
# -----------------------------

total_transactions = len(df)
normal_transactions = (df["Class"] == 0).sum()
fraud_transactions = (df["Class"] == 1).sum()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Transactions",
        f"{total_transactions:,}"
    )

with col2:
    st.metric(
        "Normal Transactions",
        f"{normal_transactions:,}"
    )

with col3:
    st.metric(
        "Known Fraud",
        f"{fraud_transactions:,}"
    )

with col4:
    fraud_rate = (fraud_transactions / total_transactions) * 100

    st.metric(
        "Fraud Rate",
        f"{fraud_rate:.3f}%"
    )

st.divider()

# -----------------------------
# TRANSACTION INPUT
# -----------------------------

st.header("🔍 Transaction Analysis")

st.write(
    "Enter the transaction information below to check "
    "whether it appears anomalous."
)

col1, col2 = st.columns(2)

with col1:

    transaction_time = st.number_input(
        "Transaction Time",
        min_value=0.0,
        value=50000.0
    )

    amount = st.number_input(
        "Transaction Amount",
        min_value=0.0,
        value=100.0
    )

with col2:

    st.info(
        "The dataset contains anonymized V1–V28 features. "
        "For the demo, representative values are used."
    )

# V1-V28 inputs
feature_values = []

for i in range(1, 29):

    value = st.number_input(
        f"V{i}",
        value=0.0
    )

    feature_values.append(value)

# -----------------------------
# PREDICTION
# -----------------------------

if st.button(
    "🚨 Analyze Transaction",
    use_container_width=True
):

    input_data = pd.DataFrame(
        [[
            transaction_time,
            *feature_values,
            amount
        ]],
        columns=[
            "Time",
            *[f"V{i}" for i in range(1, 29)],
            "Amount"
        ]
    )

    # Scale Time and Amount
    input_data[["Time", "Amount"]] = scaler.transform(
        input_data[["Time", "Amount"]]
    )

    # Prediction
    prediction = model.predict(input_data)[0]

    # Anomaly score
    score = model.decision_function(input_data)[0]

    st.divider()

    if prediction == -1:

        st.error("⚠️ SUSPICIOUS TRANSACTION")

        st.metric(
            "Anomaly Score",
            f"{score:.4f}"
        )

        st.write(
            "The transaction shows characteristics that "
            "differ significantly from normal transaction patterns."
        )

    else:

        st.success("✅ NORMAL TRANSACTION")

        st.metric(
            "Anomaly Score",
            f"{score:.4f}"
        )

        st.write(
            "The transaction does not appear anomalous "
            "according to the trained model."
        )

st.divider()

# -----------------------------
# DATA VISUALIZATION
# -----------------------------

st.header("📊 Transaction Overview")

chart_data = df["Class"].value_counts()

st.bar_chart(chart_data)