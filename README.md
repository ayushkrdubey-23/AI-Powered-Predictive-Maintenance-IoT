# AI-Powered Predictive Maintenance System for IoT Devices

An interactive machine-learning dashboard that simulates IoT sensor readings and estimates potential equipment failure risk using a Random Forest classifier.

> **Disclaimer:** This is an educational prototype built with synthetic data. It is not connected to physical IoT hardware and must not be used for real-world safety-critical maintenance decisions.

## Project Overview

Unexpected equipment failures can cause downtime, repair costs, and operational disruption. Predictive maintenance uses machine data to identify possible failure patterns before equipment breaks down.

This project demonstrates a basic predictive maintenance workflow using simulated sensor data, a machine-learning model, and an interactive Streamlit dashboard.

## Key Features

- Synthetic dataset containing 2,000 simulated sensor records.
- Temperature, vibration, pressure, and operating-hours inputs.
- Random Forest classification model.
- Interactive machine sensor controls.
- Estimated failure probability and illustrative risk status.
- Interactive sensor distribution charts.
- Dataset label distribution visualization.
- Random Forest feature importance chart.
- Model evaluation report and test predictions.
- Dark-themed dashboard interface.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data processing |
| NumPy | Synthetic data generation |
| Scikit-learn | Machine-learning model and evaluation |
| Joblib | Saving and loading the trained model |
| Streamlit | Interactive dashboard |
| Plotly | Interactive visualizations |
| Matplotlib | Dataset generation and analysis support |

## Project Structure

```text
AI-Predictive-Maintenance-IoT/
├── data/
│   ├── generate_dataset.py
│   └── iot_sensor_data.csv
├── src/
│   └── train_model.py
├── models/
│   └── maintenance_model.joblib
├── outputs/
│   ├── evaluation.txt
│   └── predictions.csv
├── images/
├── docs/
├── main.py
├── main_backup.py
├── requirements.txt
└── README.md
```

## How It Works

1. Generate synthetic IoT sensor readings.
2. Prepare sensor features and failure labels.
3. Split the data into training and testing sets.
4. Train a Random Forest classifier.
5. Evaluate predictions on the test dataset.
6. Save the trained model and evaluation outputs.
7. Run the Streamlit dashboard.
8. Adjust virtual sensor readings to explore model predictions.

## Model Evaluation

The current synthetic-data experiment produced the following results:

| Metric | Result |
|---|---:|
| Accuracy | 69.50% |
| Precision | 47.62% |
| Recall | 42.74% |

These metrics describe the current experiment, not guaranteed performance on real industrial machines. The model was trained using simulated labels and requires validation on representative real-world data before practical use.

## Installation and Usage

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd AI-Predictive-Maintenance-IoT
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your actual repository URL.

### 2. Create and activate a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Generate the dataset

```powershell
python data/generate_dataset.py
```

### 5. Train the model

```powershell
python src/train_model.py
```

### 6. Launch the dashboard

```powershell
python -m streamlit run main.py
```

Open the local URL shown in the terminal, usually `http://localhost:8501`.

## Limitations and Future Improvements

- Replace synthetic data with a suitable, documented real-world dataset.
- Validate the model using cross-validation and appropriate failure-detection metrics.
- Connect actual IoT sensors or a simulated streaming data source.
- Add an API for predictions and integration with other applications.
- Add maintenance history, alert logging, and model monitoring.
- Evaluate false negatives carefully before considering operational deployment.

## Learning Outcomes

- Synthetic dataset generation and data preparation.
- Supervised machine-learning classification.
- Model evaluation using precision, recall, and accuracy.
- Model persistence with Joblib.
- Interactive dashboard development.
- Data visualization and project documentation.

## Data Notice

All generated readings and failure labels in this project are simulated. The displayed failure probability is a model estimate, not a confirmed probability of real equipment failure.

## Author

Developed as a student portfolio project to demonstrate Python, machine learning, data visualization, and dashboard development.