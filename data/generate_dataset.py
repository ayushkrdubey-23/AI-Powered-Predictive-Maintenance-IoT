
import os
import numpy as np
import pandas as pd

# Generate reproducible simulated IoT sensor data
np.random.seed(42)

TOTAL_ROWS = 2000
os.makedirs("data", exist_ok=True)

# Simulated sensor readings
temperature = np.random.normal(65, 12, TOTAL_ROWS)
vibration = np.random.normal(4, 1.8, TOTAL_ROWS)
pressure = np.random.normal(100, 12, TOTAL_ROWS)
operating_hours = np.random.randint(100, 10000, TOTAL_ROWS)
machine_id = np.random.choice(
    ["MACHINE-01", "MACHINE-02", "MACHINE-03"],
    TOTAL_ROWS
)

# Keep readings within plausible simulation ranges
temperature = np.clip(temperature, 25, 120)
vibration = np.clip(vibration, 0.1, 12)
pressure = np.clip(pressure, 50, 150)

# Simulated failure logic:
# Higher temperature/vibration and unusual pressure
# increase the chance of a failure label.
risk_score = (
    0.04 * (temperature - 60)
    + 0.35 * (vibration - 3)
    + 0.025 * np.abs(pressure - 100)
    + 0.00003 * operating_hours
)

failure_probability = 1 / (1 + np.exp(-risk_score + 2.0))
failure = np.random.binomial(1, failure_probability)

# Create the dataset
df = pd.DataFrame({
    "machine_id": machine_id,
    "temperature": np.round(temperature, 2),
    "vibration": np.round(vibration, 2),
    "pressure": np.round(pressure, 2),
    "operating_hours": operating_hours,
    "failure": failure
})

# Save dataset
output_path = "data/iot_sensor_data.csv"
df.to_csv(output_path, index=False)

print("Dataset generated successfully!")
print(f"File: {output_path}")
print(f"Total records: {len(df)}")
print("\nFirst 5 records:")
print(df.head().to_string(index=False))
print("\nFailure distribution:")
print(df["failure"].value_counts().sort_index())
print("\nDataset label: SYNTHETIC / SIMULATED DATA")
