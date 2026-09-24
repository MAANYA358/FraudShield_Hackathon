import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

# Load dataset
df = pd.read_csv("data/creditcard.csv")

# Separate features
X = df.drop("Class", axis=1)

# Scale Time and Amount
scaler = StandardScaler()

X[["Time", "Amount"]] = scaler.fit_transform(
    X[["Time", "Amount"]]
)

# Create Isolation Forest
model = IsolationForest(
    n_estimators=200,
    contamination=0.001727,
    random_state=42,
    n_jobs=-1
)

# Train
model.fit(X)

# Save model and scaler
joblib.dump(model, "models/isolation_forest.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("Model trained successfully!")
print("Model saved to models/isolation_forest.pkl")
print("Scaler saved to models/scaler.pkl")

