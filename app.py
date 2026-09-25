import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="FraudShield",
    page_icon="🛡️",
    layout="wide"
)

# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------

model = joblib.load("models/isolation_forest.pkl")
scaler = joblib.load("models/scaler.pkl")

df = pd.read_csv("data/creditcard.csv")

# ---------------------------------------------------
# HEADER
# ---------------------------------------------------

st.title("🛡️ FraudShield")
st.subheader("Real-Time Transaction Anomaly Detection")

st.write(
    "An ML-powered fraud monitoring system that detects potentially "
    "suspicious financial transactions using Isolation Forest."
)

st.divider()

# ---------------------------------------------------
# DATASET STATISTICS
# ---------------------------------------------------

total_transactions = len(df)
normal_transactions = int((df["Class"] == 0).sum())
fraud_transactions = int((df["Class"] == 1).sum())
fraud_rate = (fraud_transactions / total_transactions) * 100

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
    st.metric(
        "Fraud Rate",
        f"{fraud_rate:.3f}%"
    )

st.divider()

# ---------------------------------------------------
# LIVE MONITORING
# ---------------------------------------------------

st.header("🔴 Live Transaction Monitor")

st.write(
    "Simulate incoming transactions and automatically classify "
    "them as normal or potentially anomalous."
)

# Initialize session state
if "transactions" not in st.session_state:
    st.session_state.transactions = []

if "total_checked" not in st.session_state:
    st.session_state.total_checked = 0

if "suspicious_count" not in st.session_state:
    st.session_state.suspicious_count = 0

if "normal_count" not in st.session_state:
    st.session_state.normal_count = 0

# ---------------------------------------------------
# GENERATE TRANSACTION
# ---------------------------------------------------

if st.button(
    "🚨 Analyze New Transaction",
    use_container_width=True
):

    # Select a random transaction from the dataset
    sample = df.sample(1).iloc[0]

    transaction_time = sample["Time"]
    amount = sample["Amount"]

    # Create feature dataframe
    input_data = pd.DataFrame(
        [[
            transaction_time,
            *[sample[f"V{i}"] for i in range(1, 29)],
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

    # Convert prediction
    if prediction == -1:
        status = "SUSPICIOUS"
        st.session_state.suspicious_count += 1
    else:
        status = "NORMAL"
        st.session_state.normal_count += 1

    st.session_state.total_checked += 1

    # Actual class from dataset
    actual = "FRAUD" if sample["Class"] == 1 else "NORMAL"

    # Store result
    transaction = {
        "Transaction ID": st.session_state.total_checked,
        "Amount": round(amount, 2),
        "Anomaly Score": round(score, 4),
        "Prediction": status,
        "Actual": actual
    }

    st.session_state.transactions.insert(0, transaction)

# ---------------------------------------------------
# LIVE RESULT
# ---------------------------------------------------

if st.session_state.transactions:

    latest = st.session_state.transactions[0]

    st.subheader("Latest Transaction")

    result_col1, result_col2, result_col3, result_col4 = st.columns(4)

    with result_col1:
        st.metric(
            "Transaction ID",
            latest["Transaction ID"]
        )

    with result_col2:
        st.metric(
            "Amount",
            f"${latest['Amount']:.2f}"
        )

    with result_col3:
        st.metric(
            "Anomaly Score",
            latest["Anomaly Score"]
        )

    with result_col4:
        if latest["Prediction"] == "SUSPICIOUS":
            st.error("⚠️ SUSPICIOUS")
        else:
            st.success("✅ NORMAL")

    if latest["Prediction"] == "SUSPICIOUS":

        st.error(
            "⚠️ ALERT: Potentially anomalous transaction detected!"
        )

        st.write(
            "The transaction shows characteristics that differ "
            "from normal transaction patterns."
        )

    else:

        st.success(
            "✅ Transaction appears normal."
        )

# ---------------------------------------------------
# SESSION STATISTICS
# ---------------------------------------------------

st.divider()

st.header("📊 Session Monitoring")

session_col1, session_col2, session_col3 = st.columns(3)

with session_col1:
    st.metric(
        "Transactions Checked",
        st.session_state.total_checked
    )

with session_col2:
    st.metric(
        "Suspicious Detected",
        st.session_state.suspicious_count
    )

with session_col3:
    st.metric(
        "Normal Detected",
        st.session_state.normal_count
    )

# ---------------------------------------------------
# RECENT TRANSACTIONS
# ---------------------------------------------------

if st.session_state.transactions:

    st.divider()

    st.header("📋 Recent Transactions")

    transaction_df = pd.DataFrame(
        st.session_state.transactions
    )

    st.dataframe(
        transaction_df,
        use_container_width=True,
        hide_index=True
    )

# ---------------------------------------------------
# ANOMALY DISTRIBUTION
# ---------------------------------------------------

if st.session_state.transactions:

    st.divider()

    st.header("📈 Anomaly Monitoring")

    chart_df = pd.DataFrame(
        {
            "Normal": [
                st.session_state.normal_count
            ],
            "Suspicious": [
                st.session_state.suspicious_count
            ]
        }
    )

    st.bar_chart(chart_df)

# ---------------------------------------------------
# DATASET OVERVIEW
# ---------------------------------------------------

st.divider()

st.header("📊 Dataset Overview")

overview_col1, overview_col2 = st.columns(2)

with overview_col1:

    class_counts = df["Class"].value_counts()

    chart_data = pd.DataFrame(
        {
            "Transactions": [
                class_counts.get(0, 0),
                class_counts.get(1, 0)
            ]
        },
        index=["Normal", "Fraud"]
    )

    st.bar_chart(chart_data)

with overview_col2:

    st.write("### Model Information")

    st.write("**Algorithm:** Isolation Forest")
    st.write("**Learning Type:** Unsupervised Anomaly Detection")
    st.write("**Trees:** 200")
    st.write("**Features:** Time + V1–V28 + Amount")
    st.write("**Detection:** Anomaly Score")

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------

st.divider()

st.caption(
    "FraudShield | Machine Learning Based Fraud & Anomaly Detection"
)