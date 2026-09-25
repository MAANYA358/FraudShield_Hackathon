from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained model and scaler
model = joblib.load("models/isolation_forest.pkl")
scaler = joblib.load("models/scaler.pkl")


@app.route("/")
def home():
    return jsonify({
        "message": "FraudShield API is running",
        "endpoint": "/predict",
        "method": "POST"
    })


@app.route("/predict", methods=["POST"])
def predict():

    try:
        data = request.get_json()

        # Required columns
        required_features = [
            "Time",
            *[f"V{i}" for i in range(1, 29)],
            "Amount"
        ]

        # Check missing fields
        missing_features = [
            feature
            for feature in required_features
            if feature not in data
        ]

        if missing_features:
            return jsonify({
                "error": "Missing required features",
                "missing_features": missing_features
            }), 400

        # Create dataframe
        input_data = pd.DataFrame(
            [[data[feature] for feature in required_features]],
            columns=required_features
        )

        # Scale Time and Amount
        input_data[["Time", "Amount"]] = scaler.transform(
            input_data[["Time", "Amount"]]
        )

        # Model prediction
        prediction = model.predict(input_data)[0]

        # Anomaly score
        anomaly_score = model.decision_function(input_data)[0]

        # Convert Isolation Forest output
        if prediction == -1:
            result = "SUSPICIOUS"
        else:
            result = "NORMAL"

        return jsonify({
            "prediction": result,
            "anomaly_score": round(float(anomaly_score), 4),
            "message": (
                "Potentially anomalous transaction detected"
                if result == "SUSPICIOUS"
                else "Transaction appears normal"
            )
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )