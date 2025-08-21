# EnergySense: AI-Powered Short-Term Load Forecasting & Anomaly Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org)

## 📌 Problem Statement

Accurate short-term load forecasting is **critical for decarbonizing buildings**, which contribute nearly **one-third of global energy use and emissions**.  

This project addresses **forecasting and anomaly detection** for energy consumption in **commercial and residential buildings**, enabling:  
- Efficient **energy management**  
- Seamless **renewable integration**  
- Reduced reliance on **fossil fuels**  

By leveraging **pre-trained Time Series Foundation Models (TSFMs)**, EnergySense learns **generalizable temporal patterns** across domains, enabling accurate forecasting and anomaly detection for **new/unseen buildings without retraining**.  

Fine-tuning is also supported for **specific building types** or **operational needs**.

## 🎯 Objectives

1. Deliver **short-term energy load forecasts** (1–72 hours).  
2. Detect **abnormal energy usage patterns** (e.g., faults, inefficiencies).  
3. Provide an **interactive web application** for building operators.  
4. Enable **zero-shot inference** with pre-trained TSFMs.  
5. Support **fine-tuning** for building-specific performance.  

## 🧩 Core Features

| Feature | Description |
|---------|-------------|
| **Load Forecasting** | Predict next 1–72 hours of energy use using TSFMs. |
| **Anomaly Detection** | Highlight unusual usage patterns in consumption. |
| **Zero-Shot Forecasting** | Apply pre-trained TSFMs directly to new/unseen data. |
| **Fine-Tuning Mode** | Train model with building-specific datasets. |
| **Visualisation Dashboard** | Interactive charts for history, forecasts and anomalies. |
| **Building Manager** | Add, edit, and manage building profiles. |
| **Renewable Readiness Score** | Suggest solar/wind adoption potential. |

## 🖼️ User Interface

### 🔹 Home Dashboard
- Overview + sustainability message.  
- Key stats: average load, CO₂ savings, accuracy.  
- Quick buttons: **Start Forecasting**, **Detect Anomalies**.  

### 🔹 Forecasting Panel
- Input: building, forecast horizon (1–72h).  
- Output: forecast chart + confidence intervals.  
- CSV download option.  
- Model info shown (e.g., *TSMixer-Zero*).  

### 🔹 Anomaly Detection Panel
- Upload or select time range.  
- Output: anomalies highlighted + severity table.  
- Suggested causes (e.g., *HVAC overnight*).  

### 🔹 Fine-Tuning Panel
- Upload CSV: `timestamp, energy_kW, temperature, occupancy`.  
- Configure hyperparameters (epochs, lr, batch size).  
- Training progress + error metrics (MAE, RMSE).  
- Save fine-tuned model.  

### 🔹 Building Manager
- Add or edit building entries.  
- List of saved buildings (type, location, last update).  

### 🔹 Insights & Recommendations
- Peak reduction strategies.  
- Renewable adoption tips.  
- Exportable **PDF reports**.  

## 📋 Sidebar Navigation

```python
menu = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "📊 Forecast", "⚠️ Anomalies", "🔧 Fine-Tune", "🏢 Buildings", "💡 Insights"]
)

if menu in ["📊 Forecast", "⚠️ Anomalies"]:
    model_choice = st.sidebar.selectbox(
        "Model",
        ["TSMixer-Zero", "TimesNet-Finetuned", "Informer"]
    )
````

## ⚙️ Working Principles

### 🔹 Time Series Foundation Models (TSFMs)

* Pre-trained on **diverse datasets** (e.g., PecanStreet, BEETLE).
* Capture **daily, seasonal, occupancy-driven cycles**.
* Architectures include:

  * **TSMixer** – efficient MLP-based
  * **Informer** – Transformer-based
  * **TimesNet** – multi-period pattern learning

### 🔹 Zero-Shot Forecasting

* Input: 7 days of building load data.
* Model outputs 1–72h forecast **without retraining**.

### 🔹 Anomaly Detection Pipeline

```python
def detect_anomalies(data, model):
    reconstruction = model.autoencode(data)
    residual = abs(data - reconstruction)
    z_scores = (residual - mean) / std
    return z_scores > threshold  # e.g., z > 3.0 flagged as anomaly
```

### 🔹 Fine-Tuning Workflow

* Lightweight fine-tuning via **LoRA**.
* Saves models locally: `./models/fine_tuned_{building_id}.pt`.

### 🔹 Data Flow

```
Upload → Preprocess → TSFM Inference → Visualization → Alerts/Reports
```

## 🎨 UI Design Guidelines

* **Colors**: Green (#2E8B57), Blue (#1E90FF), Light Gray background
* **Font**: Sans-serif (default)
* **Icons**: Emojis/FontAwesome
* **Responsive**: Optimized for desktop & tablet

## 🚀 Example User Journey

1. Open app → Home Dashboard.
2. Add new building via **Building Manager**.
3. Forecast 48h horizon → observe peak loads.
4. Run **Anomaly Detection** → detect HVAC spike at 3 AM.
5. Fine-tune with local dataset → improved RMSE.
6. Export **Insights Report** for sustainability audit.

## 📦 Tech Stack

* **Frontend**: Streamlit
* **Backend**: Python, PyTorch, sktime, pandas
* **Models**: Pre-trained TSFMs (`.pt` or Hugging Face)
* **Storage**: Local (`./data`, `./models`) or optional cloud

## ✅ Future Enhancements

* Live **smart meter API** integration.
* **Multi-building dashboards**.
* **Carbon intensity overlays** with grid emissions.
* **Automated alert system** (email/SMS for anomalies).

## 📄 License

This project is licensed under the **MIT License** – see [LICENSE](LICENSE).

> 💡 *EnergySense supports smarter, greener buildings — one forecast at a time.*
