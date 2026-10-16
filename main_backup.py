
import os
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Predictive Maintenance | IoT",
    page_icon="⚙️",
    layout="wide"
)

st.title("⚙️ AI-Powered Predictive Maintenance")
st.caption("IoT Machine Health Monitoring | SYNTHETIC / DEMO ONLY")

MODEL_PATH = "models/maintenance_model.joblib"
DATA_PATH = "data/iot_sensor_data.csv"
PREDICTIONS_PATH = "outputs/predictions.csv"

if not os.path.exists(MODEL_PATH):
    st.error("Trained model not found. Run: python src\\train_model.py")
    st.stop()

if not os.path.exists(DATA_PATH):
    st.error("Sensor dataset not found. Generate the dataset first.")
    st.stop()

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
features = bundle["features"]

df = pd.read_csv(DATA_PATH)

st.sidebar.header("Virtual Sensor Controls")
st.sidebar.caption("Change readings to simulate machine conditions.")

temperature = st.sidebar.slider(
    "Temperature (°C)", 25.0, 120.0, 65.0, 1.0
)
vibration = st.sidebar.slider(
    "Vibration", 0.1, 12.0, 4.0, 0.1
)
pressure = st.sidebar.slider(
    "Pressure", 50.0, 150.0, 100.0, 1.0
)
operating_hours = st.sidebar.slider(
    "Operating hours", 100, 10000, 3000, 100
)

sensor_values = pd.DataFrame([{
    "temperature": temperature,
    "vibration": vibration,
    "pressure": pressure,
    "operating_hours": operating_hours
}])[features]

failure_prediction = int(model.predict(sensor_values)[0])
failure_probability = float(model.predict_proba(sensor_values)[0][1])

st.subheader("Machine Health Overview")

total_records = len(df)
failure_records = int(df["failure"].sum())
normal_records = total_records - failure_records

col1, col2, col3, col4 = st.columns(4)

col1.metric("Sensor Records", f"{total_records:,}")
col2.metric("Simulated Failures", failure_records)
col3.metric("Normal Records", normal_records)
col4.metric("Predicted Failure Risk", f"{failure_probability:.1%}")

st.divider()

left, right = st.columns(2)

with left:
    st.subheader("Live Virtual Sensor Readings")
    st.write(f"**Temperature:** {temperature:.1f} °C")
    st.write(f"**Vibration:** {vibration:.1f}")
    st.write(f"**Pressure:** {pressure:.1f}")
    st.write(f"**Operating Hours:** {operating_hours:,}")

with right:
    st.subheader("AI Prediction")

    if failure_prediction == 1:
        st.error("⚠️ POTENTIAL FAILURE RISK DETECTED")
    else:
        st.success("✅ NORMAL — Model predicts no failure")

    st.progress(failure_probability)
    st.write(f"Estimated failure probability: {failure_probability:.1%}")
    st.caption(
        "Prediction from a model trained on synthetic labels. "
        "Not a validated industrial warning."
    )

st.divider()

st.subheader("Sensor Data Analysis")

chart_choice = st.selectbox(
    "Choose sensor",
    ["temperature", "vibration", "pressure", "operating_hours"]
)

fig, ax = plt.subplots(figsize=(9, 4))
ax.hist(df[chart_choice], bins=30, edgecolor="black")
ax.set_title(f"Distribution of {chart_choice}")
ax.set_xlabel(chart_choice)
ax.set_ylabel("Number of records")
st.pyplot(fig)
plt.close(fig)

left_chart, right_chart = st.columns(2)

with left_chart:
    st.subheader("Failure vs Normal")
    counts = df["failure"].value_counts().reindex(
        [0, 1], fill_value=0
    )
    fig1, ax1 = plt.subplots()
    ax1.bar(["Normal", "Failure"], counts.values)
    ax1.set_ylabel("Number of records")
    st.pyplot(fig1)
    plt.close(fig1)

with right_chart:
    st.subheader("Model Evaluation")
    if os.path.exists("outputs/evaluation.txt"):
        with open("outputs/evaluation.txt", encoding="utf-8") as file:
            st.text(file.read())
    else:
        st.info("Train the model to generate evaluation results.")

st.subheader("Recent Simulated Sensor Records")
st.dataframe(df.tail(10), use_container_width=True)

if os.path.exists(PREDICTIONS_PATH):
    st.subheader("Model Test Predictions")
    prediction_df = pd.read_csv(PREDICTIONS_PATH)
    st.dataframe(prediction_df.head(10), use_container_width=True)

st.caption(
    "Educational demonstration only. All sensor data and labels are simulated."
)
