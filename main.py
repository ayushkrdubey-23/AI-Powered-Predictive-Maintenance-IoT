
import os
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="IoT Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- THEME --------------------
st.markdown("""
<style>
.stApp {
    background: #0b1120;
    color: #e5e7eb;
}
[data-testid="stSidebar"] {
    background: #111827;
    border-right: 1px solid #263449;
}
.block-container {
    padding-top: 1.5rem;
    max-width: 1500px;
}
h1, h2, h3 {
    color: #f8fafc !important;
}
.kpi {
    background: linear-gradient(135deg, #172338, #111827);
    border: 1px solid #29384f;
    border-radius: 12px;
    padding: 18px;
    min-height: 115px;
}
.kpi-label {
    color: #9caec5;
    font-size: 13px;
}
.kpi-value {
    color: #f8fafc;
    font-size: 27px;
    font-weight: 700;
    margin-top: 8px;
}
.kpi-note {
    color: #94a3b8;
    font-size: 12px;
}
hr {
    border-color: #263449;
}

/* Readable text selection */
::selection {
    background: #334155 !important;
    color: #ffffff !important;
}
/* Consistent dark inputs */
[data-testid="stSidebar"] label,
[data-testid="stWidgetLabel"] p {
    color: #e2e8f0 !important;
}

[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background-color: #172338 !important;
    color: #f8fafc !important;
    border-color: #475569 !important;
}

[data-testid="stSidebar"] [data-baseweb="select"] input {
    color: #f8fafc !important;
}

/* Dark tables */
[data-testid="stDataFrame"],
[data-testid="stTable"] {
    color: #e5e7eb !important;
}

[data-testid="stDataFrame"] [role="grid"] {
    background-color: #111827 !important;
}

/* Readable slider values */
[data-testid="stSlider"] {
    color: #e2e8f0 !important;
}

</style>
""", unsafe_allow_html=True)

# -------------------- HEADER --------------------
st.title("⚙️ Predictive Maintenance Center")
st.markdown(
    "Industrial IoT | Machine Health Intelligence | "
    "**SYNTHETIC / DEMO ONLY**"
)
st.divider()

MODEL_PATH = "models/maintenance_model.joblib"
DATA_PATH = "data/iot_sensor_data.csv"
EVAL_PATH = "outputs/evaluation.txt"

# -------------------- LOAD DATA --------------------
if not os.path.exists(MODEL_PATH):
    st.error("Model not found. Run: python src\\train_model.py")
    st.stop()

if not os.path.exists(DATA_PATH):
    st.error("Dataset not found. Run: python data\\generate_dataset.py")
    st.stop()

try:
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    features = bundle["features"]
    df = pd.read_csv(DATA_PATH)
except Exception as exc:
    st.error(f"Unable to load project files: {exc}")
    st.stop()

required = set(features + ["failure", "machine_id"])
if not required.issubset(df.columns):
    st.error("Dataset columns do not match the trained model.")
    st.stop()

# -------------------- SIDEBAR --------------------
st.sidebar.title("🛠️ Virtual Machine")
st.sidebar.caption("Adjust readings to simulate machine conditions.")

machine_id = st.sidebar.selectbox(
    "Machine ID",
    sorted(df["machine_id"].unique())
)
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
    "Operating Hours", 100, 10000, 3000, 100
)

sensor_input = pd.DataFrame([{
    "temperature": temperature,
    "vibration": vibration,
    "pressure": pressure,
    "operating_hours": operating_hours
}])[features]

prediction = int(model.predict(sensor_input)[0])
probability = float(model.predict_proba(sensor_input)[0][1])

if probability >= 0.70:
    status = "HIGH RISK"
    status_color = "#ef4444"
elif probability >= 0.40:
    status = "WARNING"
    status_color = "#f59e0b"
else:
    status = "NORMAL"
    status_color = "#22c55e"

total = len(df)
failures = int(df["failure"].sum())
normal = total - failures

# -------------------- KPI CARDS --------------------
st.subheader("Executive Overview")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""
    <div class="kpi">
      <div class="kpi-label">TOTAL SENSOR RECORDS</div>
      <div class="kpi-value">{total:,}</div>
      <div class="kpi-note">Simulated observations</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="kpi">
      <div class="kpi-label">FAILURE-LABELLED RECORDS</div>
      <div class="kpi-value">{failures:,}</div>
      <div class="kpi-note">Synthetic dataset labels</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="kpi">
      <div class="kpi-label">NORMAL-LABELLED RECORDS</div>
      <div class="kpi-value">{normal:,}</div>
      <div class="kpi-note">Synthetic dataset labels</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="kpi">
      <div class="kpi-label">ESTIMATED FAILURE PROBABILITY</div>
      <div class="kpi-value">{probability:.1%}</div>
      <div class="kpi-note">Virtual sensor input</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# -------------------- MACHINE STATUS --------------------
left, right = st.columns([1, 1])

with left:
    st.subheader("📡 Machine Sensor Readings")
    readings = pd.DataFrame({
        "Sensor": [
            "Temperature", "Vibration",
            "Pressure", "Operating Hours"
        ],
        "Reading": [
            f"{temperature:.1f} °C",
            f"{vibration:.1f}",
            f"{pressure:.1f}",
            f"{operating_hours:,} h"
        ]
    })
    st.dataframe(readings, hide_index=True, use_container_width=True)

    st.caption(f"Selected machine: {machine_id}")

with right:
    st.subheader("🛡️ AI Risk Assessment")
    st.markdown(
        f'<div class="kpi" style="border-left:5px solid {status_color};">'
        f'<div class="kpi-label">MODEL-BASED STATUS</div>'
        f'<div class="kpi-value" style="color:{status_color};">'
        f'{status}</div>'
        f'<div class="kpi-note">Estimated probability: {probability:.1%}</div>'
        '</div>',
        unsafe_allow_html=True
    )
    st.progress(probability)
    st.write(f"Model predicted class: **{prediction}**")
    st.caption(
        "Risk bands are illustrative thresholds. "
        "This is not a validated industrial alarm."
    )

st.divider()

# -------------------- SENSOR ANALYTICS --------------------
st.subheader("📊 Sensor Analytics")

sensor_name = st.selectbox(
    "Select sensor to analyse",
    ["temperature", "vibration", "pressure", "operating_hours"]
)

fig_hist = px.histogram(
    df,
    x=sensor_name,
    color=df["failure"].map({0: "Normal label", 1: "Failure label"}),
    nbins=30,
    barmode="overlay",
    opacity=0.75,
    title=f"{sensor_name.replace('_', ' ').title()} Distribution"
)
fig_hist.update_layout(
    template="plotly_dark",
    paper_bgcolor="#0b1120",
    plot_bgcolor="#111827",
    legend_title_text="Dataset label"
)
st.plotly_chart(fig_hist, use_container_width=True)

chart_left, chart_right = st.columns(2)

with chart_left:
    st.subheader("Dataset Label Distribution")
    label_counts = df["failure"].map({
        0: "Normal label", 1: "Failure label"
    }).value_counts().reset_index()
    label_counts.columns = ["Label", "Records"]
    fig_labels = px.pie(
        label_counts,
        names="Label",
        values="Records",
        hole=0.55,
        title="Normal vs Failure Labels"
    )
    fig_labels.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0b1120"
    )
    st.plotly_chart(fig_labels, use_container_width=True)

with chart_right:
    st.subheader("Feature Importance")
    if hasattr(model, "feature_importances_"):
        importance = pd.DataFrame({
            "Feature": features,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=True)
        fig_imp = px.bar(
            importance,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Random Forest Feature Importance"
        )
        fig_imp.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0b1120",
            plot_bgcolor="#111827"
        )
        st.plotly_chart(fig_imp, use_container_width=True)
    else:
        st.info("Feature importance is unavailable for this model.")

# -------------------- MODEL EVALUATION --------------------
st.divider()
st.subheader("🧪 Model Evaluation")

if os.path.exists(EVAL_PATH):
    with open(EVAL_PATH, encoding="utf-8") as file:
        st.code(file.read(), language="text")
else:
    st.info("Evaluation report not found. Train the model first.")

# -------------------- DATA TABLES --------------------
st.divider()
st.subheader("📋 Recent Sensor Records")
st.dataframe(df.tail(10), hide_index=True, use_container_width=True)

prediction_path = "outputs/predictions.csv"
if os.path.exists(prediction_path):
    with st.expander("View model test predictions"):
        predictions_df = pd.read_csv(prediction_path)
        st.dataframe(
            predictions_df.head(20),
            hide_index=True,
            use_container_width=True
        )

st.caption(
    "Educational prototype | Synthetic data only | "
    "Not connected to live IoT hardware | Not for safety-critical decisions"
)
