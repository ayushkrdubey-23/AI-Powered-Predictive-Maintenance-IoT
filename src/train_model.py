
import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix
)

# Project folders
os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# 1. Load simulated IoT dataset
df = pd.read_csv("data/iot_sensor_data.csv")

# 2. Select sensor features
features = [
    "temperature",
    "vibration",
    "pressure",
    "operating_hours"
]

X = df[features]
y = df["failure"]

# 3. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("\nTraining Random Forest model...")

# 4. Train the model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# 5. Make predictions
y_pred = model.predict(X_test)

# 6. Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(
    y_test, y_pred, zero_division=0
)
recall = recall_score(
    y_test, y_pred, zero_division=0
)

print("\nMODEL EVALUATION")
print("----------------")
print(f"Accuracy:  {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")

print("\nClassification Report:")
report = classification_report(
    y_test,
    y_pred,
    zero_division=0
)
print(report)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 7. Save the trained model and feature names
joblib.dump(
    {"model": model, "features": features},
    "models/maintenance_model.joblib"
)

# 8. Save evaluation results
with open("outputs/evaluation.txt", "w", encoding="utf-8") as f:
    f.write("SYNTHETIC / SIMULATED DATA\n\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n\n")
    f.write(report)
    f.write("\n\nConfusion Matrix:\n")
    f.write(str(confusion_matrix(y_test, y_pred)))

# 9. Save test predictions for the dashboard
results = X_test.copy()
results["actual_failure"] = y_test
results["predicted_failure"] = y_pred
results["failure_probability"] = model.predict_proba(X_test)[:, 1]

results.to_csv("outputs/predictions.csv", index=False)

print("\nModel saved: models/maintenance_model.joblib")
print("Evaluation saved: outputs/evaluation.txt")
print("Predictions saved: outputs/predictions.csv")
print("\nTraining completed successfully!")
