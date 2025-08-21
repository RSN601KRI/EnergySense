import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import base64
from io import BytesIO
import os
import time

# Set page config
st.set_page_config(
    page_title="EnergySense - AI for Sustainable Buildings",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "buildings" not in st.session_state:
    st.session_state.buildings = [
        {"Name": "Office Tower A", "Type": "Commercial", "Location": "New York", "Added": "2024-01-10", "Size": "50,000 sq ft"},
        {"Name": "Residential Block B", "Type": "Residential", "Location": "San Francisco", "Added": "2024-01-12", "Size": "30,000 sq ft"},
        {"Name": "Retail Mall C", "Type": "Retail", "Location": "Chicago", "Added": "2024-01-15", "Size": "75,000 sq ft"}
    ]

if "model_performance" not in st.session_state:
    st.session_state.model_performance = {
        "TSMixer-Zero": {"MAE": 0.23, "RMSE": 0.31, "Accuracy": 94.2},
        "TimesNet-Finetuned": {"MAE": 0.18, "RMSE": 0.25, "Accuracy": 96.1},
        "Informer": {"MAE": 0.28, "RMSE": 0.35, "Accuracy": 92.8}
    }

if "theme" not in st.session_state:
    st.session_state.theme = "Light"

# Theme detection for system mode
def get_system_theme():
    """Detect system theme preference (simplified version)"""
    return "Dark"  # Placeholder - in real app, you'd detect actual system theme

# Theme configurations
def get_theme_config(theme_mode):
    themes = {
        "Light": {
            "bg_primary": "#ffffff",
            "bg_secondary": "#f8f9ff",
            "text_primary": "#1E90FF",
            "text_secondary": "#222",
            "text_muted": "#555",
            "border": "#bee5eb",
            "success_bg": "#d4edda",
            "success_border": "#c3e6cb",
            "info_bg": "#d1ecf1",
            "info_border": "#bee5eb",
            "sidebar_bg": "linear-gradient(to bottom, #f0f5ff, #ffffff)",
            "metric_bg": "#f8f9ff",
            "button_bg": "#1E90FF",
            "button_hover": "#0066cc",
            "plotly_template": "plotly_white"
        },
        "Dark": {
            "bg_primary": "#0e1117",
            "bg_secondary": "#1a1d29",
            "text_primary": "#4FC3F7",
            "text_secondary": "#ffffff",
            "text_muted": "#b0b3b8",
            "border": "#2d3748",
            "success_bg": "#1a2e1a",
            "success_border": "#2d5a2d",
            "info_bg": "#1a252e",
            "info_border": "#2d4a5a",
            "sidebar_bg": "linear-gradient(to bottom, #1a1d29, #0e1117)",
            "metric_bg": "#1a1d29",
            "button_bg": "#4FC3F7",
            "button_hover": "#29B6F6",
            "plotly_template": "plotly_dark"
        },
        "System": {
            # Will be resolved to Light or Dark based on system detection
        }
    }
    
    if theme_mode == "System":
        detected_theme = get_system_theme()
        return themes[detected_theme]
    
    return themes.get(theme_mode, themes["Light"])

# Apply theme CSS
def apply_theme_css(theme_config):
    css = f"""
    <style>
    /* Import fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Root variables */
    :root {{
        --bg-primary: {theme_config['bg_primary']};
        --bg-secondary: {theme_config['bg_secondary']};
        --text-primary: {theme_config['text_primary']};
        --text-secondary: {theme_config['text_secondary']};
        --text-muted: {theme_config['text_muted']};
        --border-color: {theme_config['border']};
        --success-bg: {theme_config['success_bg']};
        --success-border: {theme_config['success_border']};
        --info-bg: {theme_config['info_bg']};
        --info-border: {theme_config['info_border']};
    }}

    /* General app styling */
    .stApp {{
        background-color: var(--bg-primary) !important;
        font-family: 'Inter', 'Segoe UI', sans-serif !important;
    }}
    
    .css-18e3th9 {{
        padding-top: 1rem;
        padding-bottom: 1rem;
    }}
    
    .css-1d391kg {{
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }}

    /* Sidebar Styling */
    .css-1lcbmhc, .css-1cypcdb, .css-17lntkn {{
        background: {theme_config['sidebar_bg']} !important;
    }}
    
    .sidebar .sidebar-content {{
        background: {theme_config['sidebar_bg']};
        color: var(--text-primary);
        font-family: 'Inter', sans-serif;
    }}
    
    .sidebar .sidebar-content h1 {{
        color: var(--text-primary) !important;
        margin: 0;
        font-size: 1.8em;
        font-weight: 700;
    }}
    
    .sidebar .sidebar-content p {{
        color: var(--text-muted) !important;
        font-size: 0.95em;
        margin: 4px 0 0 0;
    }}
    
    .css-1lcbmhc .css-1v0mbdj {{
        color: var(--text-secondary) !important;
    }}
    
    .css-1lcbmhc label {{
        color: var(--text-secondary) !important;
        font-weight: 500;
    }}

    /* Main Content */
    .main .block-container {{
        max-width: 1200px;
        padding-top: 1rem;
        padding-bottom: 3rem;
        background-color: var(--bg-primary) !important;
    }}
    
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {{
        color: var(--text-primary) !important;
        font-weight: 600 !important;
    }}
    
    .stMarkdown p, .stMarkdown li, .stMarkdown span {{
        color: var(--text-secondary) !important;
    }}
    
    /* Buttons */
    .stButton > button {{
        background-color: {theme_config['button_bg']} !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.5em 1.2em !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
    }}
    
    .stButton > button:hover {{
        background-color: {theme_config['button_hover']} !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(79, 195, 247, 0.3) !important;
    }}
    
    /* Metric cards */
    .metric-card {{
        background: var(--bg-secondary) !important;
        padding: 16px !important;
        border-radius: 12px !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1) !important;
        margin-bottom: 1rem !important;
        text-align: center !important;
        border: 1px solid var(--border-color) !important;
        transition: transform 0.2s ease !important;
    }}
    
    .metric-card:hover {{
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 16px rgba(0,0,0,0.15) !important;
    }}
    
    .metric-card b {{
        color: var(--text-primary) !important;
        font-size: 1.5em !important;
    }}
    
    /* Alert boxes */
    .success-box {{
        background: var(--success-bg) !important;
        border: 1px solid var(--success-border) !important;
        border-radius: 8px !important;
        padding: 12px !important;
        margin: 10px 0 !important;
        color: var(--text-secondary) !important;
    }}
    
    .info-box {{
        background: var(--info-bg) !important;
        border: 1px solid var(--info-border) !important;
        border-radius: 8px !important;
        padding: 12px !important;
        margin: 10px 0 !important;
        color: var(--text-secondary) !important;
    }}
    
    /* Streamlit components theming */
    .stSelectbox label, .stSlider label, .stRadio label {{
        color: var(--text-secondary) !important;
        font-weight: 500 !important;
    }}
    
    .stSelectbox > div > div {{
        background-color: var(--bg-secondary) !important;
        border-color: var(--border-color) !important;
        color: var(--text-secondary) !important;
    }}
    
    .stTextInput > div > div > input {{
        background-color: var(--bg-secondary) !important;
        color: var(--text-secondary) !important;
        border-color: var(--border-color) !important;
    }}
    
    .stNumberInput > div > div > input {{
        background-color: var(--bg-secondary) !important;
        color: var(--text-secondary) !important;
        border-color: var(--border-color) !important;
    }}
    
    /* DataFrame styling */
    .stDataFrame {{
        background-color: var(--bg-secondary) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
    }}
    
    /* Metric widget styling */
    [data-testid="metric-container"] {{
        background-color: var(--bg-secondary) !important;
        border: 1px solid var(--border-color) !important;
        padding: 12px !important;
        border-radius: 8px !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05) !important;
    }}
    
    [data-testid="metric-container"] > div > div > div > div {{
        color: var(--text-secondary) !important;
    }}
    
    /* Progress bar */
    .stProgress > div > div > div > div {{
        background-color: var(--text-primary) !important;
    }}
    
    /* Expander */
    .streamlit-expanderHeader {{
        background-color: var(--bg-secondary) !important;
        color: var(--text-secondary) !important;
        border-color: var(--border-color) !important;
    }}
    
    .streamlit-expanderContent {{
        background-color: var(--bg-secondary) !important;
        border-color: var(--border-color) !important;
    }}
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        background-color: var(--bg-secondary) !important;
        color: var(--text-secondary) !important;
        border-radius: 8px !important;
        padding: 8px 16px !important;
        border: 1px solid var(--border-color) !important;
    }}
    
    .stTabs [aria-selected="true"] {{
        background-color: var(--text-primary) !important;
        color: white !important;
    }}
    
    /* File uploader */
    .stFileUploader > label {{
        color: var(--text-secondary) !important;
    }}
    
    .stFileUploader [data-testid="stFileUploaderDropzone"] {{
        background-color: var(--bg-secondary) !important;
        border-color: var(--border-color) !important;
        color: var(--text-secondary) !important;
    }}
    
    /* Form styling */
    .stForm {{
        background-color: var(--bg-secondary) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 12px !important;
        padding: 20px !important;
    }}
    
    /* Sidebar theme selector highlight */
    .theme-selector {{
        background-color: var(--bg-secondary) !important;
        border: 2px solid var(--text-primary) !important;
        border-radius: 8px !important;
        padding: 8px !important;
        margin: 8px 0 !important;
    }}
    </style>
    """
    return css

# -----------------------------
# 🖼️ Sidebar: Logo, Title & Navigation
# -----------------------------
st.sidebar.markdown(
    """
    <div style="padding: 20px 15px; text-align: center;">
        <h1>⚡ EnergySense</h1>
        <p style="font-size: 0.9em; margin-top: -10px;">
            AI for Sustainable Buildings
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("---")

# Theme selector with enhanced UI
st.sidebar.markdown("### 🎨 Theme")
theme_options = ["Light", "Dark", "System"]
theme_icons = {"Light": "☀️", "Dark": "🌙", "System": "⚙️"}

# Create custom theme selector
selected_theme = st.sidebar.selectbox(
    "Choose theme mode",
    theme_options,
    index=theme_options.index(st.session_state.theme),
    format_func=lambda x: f"{theme_icons[x]} {x}",
    key="theme_selector"
)

# Update theme if changed
if selected_theme != st.session_state.theme:
    st.session_state.theme = selected_theme
    st.rerun()

# Apply theme
current_theme = st.session_state.theme
if current_theme == "System":
    # For demo purposes, we'll use Dark. In production, detect actual system theme
    effective_theme = "Dark"
    st.sidebar.info("🔄 Auto-detected: Dark mode")
else:
    effective_theme = current_theme

theme_config = get_theme_config(effective_theme)
st.markdown(apply_theme_css(theme_config), unsafe_allow_html=True)

# Update plotly template for charts
plotly_template = theme_config['plotly_template']

# Navigation
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "📊 Forecast", "⚠️ Anomalies", "🔧 Fine-Tune", "🏢 Buildings", "💡 Insights"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")

# Real-time system status
st.sidebar.markdown("### 📊 System Status")
st.sidebar.metric("Active Buildings", len(st.session_state.buildings))
st.sidebar.metric("Models Available", 3)
st.sidebar.metric("Avg. Accuracy", "94.4%")

# Theme preview
st.sidebar.markdown("---")
st.sidebar.markdown("### 🎨 Theme Preview")
preview_html = f"""
<div style="
    background: {theme_config['bg_secondary']}; 
    padding: 12px; 
    border-radius: 8px; 
    border: 1px solid {theme_config['border']};
    text-align: center;
    margin: 8px 0;
">
    <div style="color: {theme_config['text_primary']}; font-weight: bold; font-size: 1.1em;">
        Current Theme
    </div>
    <div style="color: {theme_config['text_secondary']}; font-size: 0.9em; margin-top: 4px;">
        {effective_theme} Mode Active
    </div>
</div>
"""
st.sidebar.markdown(preview_html, unsafe_allow_html=True)

# -----------------------------
# 🏠 PAGE 1: Home
# -----------------------------
if page == "🏠 Home":
    st.title("Welcome to EnergySense ⚡")
    st.markdown("""
    ### **AI-Powered Load Forecasting & Anomaly Detection**
    Optimize energy use in buildings using **Time Series Foundation Models (TSFMs)** to support decarbonization and renewable integration.
    """)

    # Hero section with columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        **🚀 Key Features:**
        - 🔮 Zero-shot forecasting for new buildings
        - 🚨 Real-time anomaly detection
        - 🔧 Fine-tune models for higher accuracy
        - 🌱 Reduce fossil fuel reliance
        - 📊 Interactive visualizations
        - 🏢 Multi-building management
        """)
        
        # Quick action buttons
        st.markdown("### 🎯 Quick Actions")
        button_col1, button_col2, button_col3 = st.columns(3)
        with button_col1:
            if st.button("📊 Run Demo Forecast"):
                st.session_state.demo_forecast = True
                st.success("Demo forecast ready! Go to Forecast page.")
        with button_col2:
            if st.button("🔍 Detect Anomalies"):
                st.session_state.demo_anomaly = True
                st.success("Sample data loaded! Go to Anomalies page.")
        with button_col3:
            if st.button("🏢 Add Building"):
                st.session_state.quick_add = True
                st.success("Quick add mode! Go to Buildings page.")

    with col2:
        # Display a sample chart
        st.markdown("### 📈 Live Energy Data")
        dates = pd.date_range(start='2024-01-01', periods=24, freq='H')
        energy_data = 50 + 20 * np.sin(np.arange(24) / 24 * 2 * np.pi * 2) + np.random.normal(0, 5, 24)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates, y=energy_data, mode='lines+markers',
            name='Energy Consumption', line=dict(color='#1E90FF', width=3)
        ))
        fig.update_layout(
            title="Sample Building - Last 24 Hours",
            xaxis_title="Time",
            yaxis_title="Energy (kW)",
            height=300,
            margin=dict(l=0, r=0, t=30, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)

    # Key Metrics
    st.markdown("### 📊 Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card"><b>100+</b><br>Buildings Supported</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><b>94.4%</b><br>Avg. Accuracy</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><b>12.5 T</b><br>CO₂ Saved/Year</div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card"><b>24/7</b><br>Monitoring</div>', unsafe_allow_html=True)

    # Recent Activity
    st.markdown("### 📋 Recent Activity")
    recent_activities = [
        "🔮 Forecast generated for Office Tower A - 99.2% confidence",
        "⚠️ Anomaly detected in Retail Mall C - High consumption at 3 AM",
        "🔧 Model fine-tuning completed for Residential Block B",
        "📊 Weekly energy report generated for all buildings"
    ]
    
    for activity in recent_activities:
        st.info(activity)

# -----------------------------
# 📊 PAGE 2: Forecast
# -----------------------------
elif page == "📊 Forecast":
    st.header("📊 Short-Term Load Forecasting")
    st.markdown("Generate energy consumption forecasts using Time Series Foundation Models")

    # Input controls
    col1, col2, col3 = st.columns(3)
    with col1:
        building = st.selectbox("Select Building", 
                               [b["Name"] for b in st.session_state.buildings],
                               help="Choose a building to forecast")
    with col2:
        horizon = st.slider("Forecast Horizon (hours)", 1, 72, 24,
                           help="How far into the future to predict")
    with col3:
        model = st.selectbox("Model", ["TSMixer-Zero (Pre-trained)", "TimesNet-Finetuned", "Informer"],
                            help="Select the forecasting model")

    # Model info
    model_name = model.split(" ")[0]
    if model_name in st.session_state.model_performance:
        perf = st.session_state.model_performance[model_name]
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("MAE", f"{perf['MAE']} kW")
        with col2:
            st.metric("RMSE", f"{perf['RMSE']} kW")
        with col3:
            st.metric("Accuracy", f"{perf['Accuracy']}%")

    # Generate forecast
    if st.button("🔮 Generate Forecast") or st.session_state.get("demo_forecast", False):
        if st.session_state.get("demo_forecast", False):
            st.session_state.demo_forecast = False
            
        with st.spinner(f"Running {model} model for {building}..."):
            # Simulate processing time
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i in range(101):
                time.sleep(0.01)  # Small delay for realism
                progress_bar.progress(i)
                if i < 30:
                    status_text.text("Loading historical data...")
                elif i < 60:
                    status_text.text("Processing with TSFM...")
                elif i < 90:
                    status_text.text("Generating predictions...")
                else:
                    status_text.text("Finalizing forecast...")
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            # Generate realistic time series data
            now = datetime.now().replace(minute=0, second=0, microsecond=0)
            past_dates = [now - timedelta(hours=i) for i in range(168, 0, -1)]  # Last 7 days
            
            # Historical data with realistic patterns
            base_load = 50
            daily_pattern = np.array([np.sin((d.hour - 6) / 24 * 2 * np.pi) for d in past_dates])
            weekly_pattern = np.array([0.8 if d.weekday() >= 5 else 1.0 for d in past_dates])
            past_load = base_load + 20 * daily_pattern * weekly_pattern + np.random.normal(0, 8, 168)
            past_load = np.clip(past_load, 10, 120)

            # Forecast data
            forecast_dates = [now + timedelta(hours=i) for i in range(1, horizon + 1)]
            forecast_daily_pattern = np.array([np.sin((d.hour - 6) / 24 * 2 * np.pi) for d in forecast_dates])
            forecast_weekly_pattern = np.array([0.8 if d.weekday() >= 5 else 1.0 for d in forecast_dates])
            
            trend = 0.1 * np.arange(horizon)  # Slight upward trend
            forecast_load = base_load + 20 * forecast_daily_pattern * forecast_weekly_pattern + trend + np.random.normal(0, 6, horizon)
            forecast_load = np.clip(forecast_load, 10, 120)
            
            # Confidence intervals
            lower = forecast_load * 0.85
            upper = forecast_load * 1.15

            # Create interactive plot
            fig = go.Figure()
            
            # Historical data
            fig.add_trace(go.Scatter(
                x=past_dates, y=past_load, 
                mode='lines', name='Historical Load',
                line=dict(color='#1E90FF', width=2),
                hovertemplate='Time: %{x}<br>Load: %{y:.1f} kW<extra></extra>'
            ))
            
            # Forecast data
            fig.add_trace(go.Scatter(
                x=forecast_dates, y=forecast_load,
                mode='lines', name='Forecast',
                line=dict(color='#FF6B35', width=2, dash='dot'),
                hovertemplate='Time: %{x}<br>Forecast: %{y:.1f} kW<extra></extra>'
            ))
            
            # Confidence interval
            fig.add_trace(go.Scatter(
                x=np.concatenate([forecast_dates, forecast_dates[::-1]]),
                y=np.concatenate([upper, lower[::-1]]),
                fill='toself', fillcolor='rgba(255, 107, 53, 0.2)',
                line=dict(color='rgba(255,107,53,0)'),
                showlegend=True, name='Confidence Interval',
                hoverinfo='skip'
            ))

            fig.update_layout(
                title=f"Energy Forecast — {building} | Model: {model_name} | Horizon: {horizon}h",
                xaxis_title="Time",
                yaxis_title="Load (kW)",
                hovermode="x unified",
                height=600,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)

            # Forecast statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Peak Load", f"{np.max(forecast_load):.1f} kW", f"{np.max(forecast_load) - base_load:.1f}")
            with col2:
                st.metric("Min Load", f"{np.min(forecast_load):.1f} kW", f"{np.min(forecast_load) - base_load:.1f}")
            with col3:
                st.metric("Avg Load", f"{np.mean(forecast_load):.1f} kW", f"{np.mean(forecast_load) - base_load:.1f}")
            with col4:
                st.metric("Total Energy", f"{np.sum(forecast_load):.0f} kWh", f"{np.sum(forecast_load) - base_load * horizon:.0f}")

            # Downloadable forecast data
            df_forecast = pd.DataFrame({
                "timestamp": forecast_dates,
                "forecast_kW": np.round(forecast_load, 2),
                "lower_bound": np.round(lower, 2),
                "upper_bound": np.round(upper, 2),
                "building": building,
                "model": model_name
            })
            
            csv = df_forecast.to_csv(index=False)
            st.download_button(
                "📥 Download Forecast (CSV)",
                data=csv,
                file_name=f"forecast_{building.replace(' ', '_')}_{horizon}h_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
            
            # Success message
            st.success(f"✅ Forecast generated successfully! {len(forecast_dates)} data points predicted with {perf['Accuracy']:.1f}% model accuracy.")

# -----------------------------
# ⚠️ PAGE 3: Anomalies
# -----------------------------
elif page == "⚠️ Anomalies":
    st.header("⚠️ Anomaly Detection")
    st.markdown("Detect abnormal energy consumption patterns using statistical and ML methods")

    # Anomaly detection methods
    method = st.selectbox("Detection Method", 
                         ["Statistical (Z-Score)", "Isolation Forest", "LSTM Autoencoder"],
                         help="Choose anomaly detection algorithm")
    
    threshold = st.slider("Sensitivity Threshold", 1.5, 4.0, 2.5, 0.1,
                         help="Lower values = more sensitive detection")

    # File upload or sample data
    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded_file = st.file_uploader("Upload Energy Data (CSV)", type=["csv"], 
                                        help="CSV should have 'timestamp' and 'energy_kW' columns")
    with col2:
        use_sample = st.button("📊 Use Sample Data") or st.session_state.get("demo_anomaly", False)
        
    if st.session_state.get("demo_anomaly", False):
        st.session_state.demo_anomaly = False
        use_sample = True

    # Process data
    df = None
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            if "timestamp" not in df.columns or "energy_kW" not in df.columns:
                st.error("❌ CSV must contain 'timestamp' and 'energy_kW' columns.")
                st.info("Expected format: timestamp, energy_kW")
            else:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df = df.sort_values("timestamp").reset_index(drop=True)
                st.success(f"✅ Uploaded data with {len(df)} records")
        except Exception as e:
            st.error(f"❌ Error reading file: {e}")
    elif use_sample:
    # Generate sample data with anomalies
        with st.spinner("Generating sample data with anomalies..."):
            dates = pd.date_range("2024-01-01", periods=500, freq='H')
        
        # Base consumption pattern (force NumPy array)
        base = 45 + 25 * np.sin((dates.hour.to_numpy() - 6) / 24 * 2 * np.pi)
        
        # Add weekly patterns (weekends different, ensure array)
        weekend_factor = np.where(dates.weekday >= 5, 0.7, 1.0).astype(float)
        base = base * weekend_factor
        
        # Add noise
        noise = np.random.normal(0, 8, len(dates))
        load = (base + noise).astype(float)   # make sure it's mutable
        
        # Inject specific anomalies
        anomaly_points = [50, 120, 200, 350, 420]
        anomaly_types = ['spike', 'drop', 'spike', 'sustained_high', 'drop']
        
        for i, (point, atype) in enumerate(zip(anomaly_points, anomaly_types)):
            if atype == 'spike':
                load[point] *= 2.5 + np.random.random()
            elif atype == 'drop':
                load[point] *= 0.3 + np.random.random() * 0.2
            elif atype == 'sustained_high':
                for j in range(5):  # 5-hour sustained anomaly
                    if point + j < len(load):
                        load[point + j] *= 1.8 + np.random.random() * 0.3
        
        load = np.clip(load, 5, 200)  # Ensure reasonable bounds
        
        df = pd.DataFrame({"timestamp": dates, "energy_kW": load})
        st.info(f"📊 Generated sample data with {len(df)} records and {len(anomaly_points)} injected anomalies")

    if df is not None:
        # Anomaly detection
        if method == "Statistical (Z-Score)":
            # Z-score based detection
            mean_val = df['energy_kW'].mean()
            std_val = df['energy_kW'].std()
            df['z_score'] = np.abs((df['energy_kW'] - mean_val) / std_val)
            anomalies = df[df['z_score'] > threshold].copy()
            
        elif method == "Isolation Forest":
            from sklearn.ensemble import IsolationForest
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            df['anomaly'] = iso_forest.fit_predict(df[['energy_kW']].values)
            anomalies = df[df['anomaly'] == -1].copy()
            
        else:  # LSTM Autoencoder (simulated)
            # Simulate autoencoder results
            reconstruction_error = np.random.exponential(1, len(df))
            # Make some points have higher reconstruction error
            high_error_indices = np.random.choice(len(df), size=int(len(df) * 0.05), replace=False)
            reconstruction_error[high_error_indices] *= 5
            df['reconstruction_error'] = reconstruction_error
            anomalies = df[df['reconstruction_error'] > threshold].copy()

        # Create visualization
        fig = go.Figure()
        
        # Normal data points
        fig.add_trace(go.Scatter(
            x=df['timestamp'], y=df['energy_kW'],
            mode='lines', name='Energy Load',
            line=dict(color='#1E90FF', width=1.5),
            hovertemplate='Time: %{x}<br>Load: %{y:.1f} kW<extra></extra>'
        ))
        
        # Anomaly points
        if len(anomalies) > 0:
            fig.add_trace(go.Scatter(
                x=anomalies['timestamp'], y=anomalies['energy_kW'],
                mode='markers', name='Anomalies',
                marker=dict(color='red', size=8, symbol='x'),
                hovertemplate='Time: %{x}<br>Anomalous Load: %{y:.1f} kW<extra></extra>'
            ))
        
        fig.update_layout(
            title=f"Anomaly Detection Results - {method}",
            xaxis_title="Time",
            yaxis_title="Energy (kW)",
            height=500,
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True)

        # Results summary
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            st.metric("Anomalies Found", len(anomalies), 
                     f"{len(anomalies)/len(df)*100:.1f}%")
        with col3:
            st.metric("Avg Energy", f"{df['energy_kW'].mean():.1f} kW")
        with col4:
            st.metric("Max Energy", f"{df['energy_kW'].max():.1f} kW")

        # Anomaly details
        if len(anomalies) > 0:
            st.markdown("### 🔍 Anomaly Details")
            
            # Add severity classification
            anomalies = anomalies.copy()
            if method == "Statistical (Z-Score)":
                anomalies['severity'] = pd.cut(anomalies['z_score'], 
                                             bins=[0, 3, 4, np.inf], 
                                             labels=['Mild', 'Moderate', 'Severe'])
            else:
                anomalies['severity'] = 'Detected'
            
            # Display anomalies table
            display_df = anomalies[['timestamp', 'energy_kW', 'severity']].copy()
            display_df.columns = ['Timestamp', 'Energy (kW)', 'Severity']
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            # Download anomalies
            csv_anomalies = anomalies.to_csv(index=False)
            st.download_button(
                "📥 Download Anomalies (CSV)",
                data=csv_anomalies,
                file_name=f"anomalies_{method.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
        else:
            st.info("✅ No significant anomalies detected with current threshold.")
            st.balloons()  # Celebration for clean data!

# -----------------------------
# 🔧 PAGE 4: Fine-Tune
# -----------------------------
elif page == "🔧 Fine-Tune":
    st.header("🔧 Model Fine-Tuning")
    st.markdown("Upload building-specific data to fine-tune Time Series Foundation Models for improved accuracy")

    # Model selection
    base_model = st.selectbox("Base Model to Fine-tune", 
                             ["TSMixer-Zero", "TimesNet", "Informer", "Chronos"],
                             help="Select pre-trained model as starting point")
    
    # Training parameters
    col1, col2, col3 = st.columns(3)
    with col1:
        epochs = st.slider("Training Epochs", 5, 100, 20)
    with col2:
        learning_rate = st.selectbox("Learning Rate", ["1e-5", "5e-5", "1e-4", "5e-4", "1e-3"], index=2)
    with col3:
        batch_size = st.selectbox("Batch Size", [16, 32, 64, 128], index=1)

    # Data upload
    st.markdown("### 📁 Training Data")
    uploaded_file = st.file_uploader("Upload Historical Energy Data (CSV)", 
                                   type=["csv"], 
                                   help="Minimum 1000 records recommended for effective fine-tuning")
    
    # Sample data option
    if st.button("📊 Generate Sample Training Data"):
        # Create comprehensive sample data
        dates = pd.date_range("2023-01-01", "2024-01-31", freq='H')
        
        # Complex realistic pattern
        base_consumption = 40
        hourly_pattern = 15 * np.sin((dates.hour - 8) / 24 * 2 * np.pi)
        daily_pattern = 5 * np.sin(dates.dayofyear / 365 * 2 * np.pi)  # Seasonal
        weekend_pattern = np.where(dates.weekday >= 5, -8, 0)  # Weekend reduction
        
        # Weather influence (simulated)
        temp_effect = 10 * np.sin(dates.dayofyear / 365 * 2 * np.pi + np.pi/2)  # Temperature effect
        
        energy = base_consumption + hourly_pattern + daily_pattern + weekend_pattern + temp_effect
        energy += np.random.normal(0, 5, len(dates))  # Noise
        energy = np.clip(energy, 5, 150)  # Reasonable bounds
        
        sample_df = pd.DataFrame({
            "timestamp": dates,
            "energy_kW": energy,
            "temperature": 20 + temp_effect + np.random.normal(0, 3, len(dates)),
            "occupancy": np.random.uniform(0.1, 1.0, len(dates))
        })
        
        st.success(f"✅ Generated {len(sample_df):,} records of sample training data")
        st.dataframe(sample_df.head(10), use_container_width=True)
        
        # Download sample data
        csv_sample = sample_df.to_csv(index=False)
        st.download_button(
            "📥 Download Sample Data",
            data=csv_sample,
            file_name=f"sample_training_data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

    # Process uploaded data
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Uploaded {len(df)} records")
            
            # Data validation
            required_cols = ["timestamp", "energy_kW"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"❌ Missing required columns: {missing_cols}")
                st.info("Required columns: timestamp, energy_kW")
            else:
                st.markdown("### 📊 Data Preview")
                st.dataframe(df.head(), use_container_width=True)
                
                # Data statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Records", f"{len(df):,}")
                with col2:
                    st.metric("Avg Energy", f"{df['energy_kW'].mean():.1f} kW")
                with col3:
                    st.metric("Peak Energy", f"{df['energy_kW'].max():.1f} kW")
                with col4:
                    st.metric("Data Quality", "95%")

                # Visualization of training data
                fig = go.Figure()
                sample_data = df.sample(min(1000, len(df))).sort_values('timestamp') if 'timestamp' in df.columns else df.sample(min(1000, len(df)))
                
                if 'timestamp' in df.columns:
                    fig.add_trace(go.Scatter(
                        x=pd.to_datetime(sample_data['timestamp']),
                        y=sample_data['energy_kW'],
                        mode='lines',
                        name='Energy Consumption',
                        line=dict(color='#1E90FF')
                    ))
                else:
                    fig.add_trace(go.Scatter(
                        y=sample_data['energy_kW'],
                        mode='lines',
                        name='Energy Consumption',
                        line=dict(color='#1E90FF')
                    ))
                
                fig.update_layout(
                    title="Training Data Overview (Sample)",
                    xaxis_title="Time" if 'timestamp' in df.columns else "Records",
                    yaxis_title="Energy (kW)",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)

                # Fine-tuning configuration
                st.markdown("### ⚙️ Fine-tuning Configuration")
                
                col1, col2 = st.columns(2)
                with col1:
                    validation_split = st.slider("Validation Split", 0.1, 0.3, 0.2)
                    early_stopping = st.checkbox("Early Stopping", value=True)
                with col2:
                    save_checkpoints = st.checkbox("Save Checkpoints", value=True)
                    use_gpu = st.checkbox("Use GPU Acceleration", value=False)

                # Start fine-tuning
                if st.button("🚀 Start Fine-Tuning"):
                    st.markdown("### 🔄 Fine-tuning Progress")
                    
                    # Progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    loss_placeholder = st.empty()
                    
                    # Simulate training process
                    train_losses = []
                    val_losses = []
                    
                    for epoch in range(epochs):
                        # Simulate training
                        time.sleep(0.1)  # Simulate processing time
                        
                        # Generate realistic loss values
                        train_loss = 1.0 * np.exp(-epoch * 0.1) + np.random.normal(0, 0.05)
                        val_loss = 1.1 * np.exp(-epoch * 0.08) + np.random.normal(0, 0.07)
                        train_losses.append(max(0.1, train_loss))
                        val_losses.append(max(0.12, val_loss))
                        
                        progress = (epoch + 1) / epochs
                        progress_bar.progress(progress)
                        status_text.text(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                        
                        # Update loss plot every few epochs
                        if epoch % 3 == 0 or epoch == epochs - 1:
                            loss_fig = go.Figure()
                            loss_fig.add_trace(go.Scatter(
                                x=list(range(1, len(train_losses) + 1)),
                                y=train_losses,
                                mode='lines',
                                name='Training Loss',
                                line=dict(color='blue')
                            ))
                            loss_fig.add_trace(go.Scatter(
                                x=list(range(1, len(val_losses) + 1)),
                                y=val_losses,
                                mode='lines',
                                name='Validation Loss',
                                line=dict(color='red')
                            ))
                            loss_fig.update_layout(
                                title="Training Progress",
                                xaxis_title="Epoch",
                                yaxis_title="Loss",
                                height=300
                            )
                            loss_placeholder.plotly_chart(loss_fig, use_container_width=True)
                    
                    # Clear progress indicators
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Final results
                    final_mae = np.random.uniform(0.15, 0.25)
                    final_rmse = final_mae * 1.3
                    improvement = np.random.uniform(15, 35)
                    
                    st.success("✅ Fine-tuning completed successfully!")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Final MAE", f"{final_mae:.3f} kW")
                    with col2:
                        st.metric("Final RMSE", f"{final_rmse:.3f} kW")
                    with col3:
                        st.metric("Improvement", f"{improvement:.1f}%", f"+{improvement:.1f}%")
                    
                    # Save model option
                    model_name = f"{base_model}_finetuned_{datetime.now().strftime('%Y%m%d_%H%M')}"
                    st.download_button(
                        "💾 Download Fine-tuned Model",
                        data=f"# Fine-tuned model weights for {model_name}\n# Model: {base_model}\n# Training Records: {len(df)}\n# Final MAE: {final_mae:.3f}\n# Training completed: {datetime.now()}",
                        file_name=f"{model_name}.pt",
                        mime="application/octet-stream"
                    )
                    
                    # Update session state with new model
                    st.session_state.model_performance[f"{base_model}-Custom"] = {
                        "MAE": final_mae,
                        "RMSE": final_rmse,
                        "Accuracy": 100 - (final_mae / df['energy_kW'].mean() * 100)
                    }
        
        except Exception as e:
            st.error(f"❌ Error processing file: {e}")
            st.info("Please ensure your CSV file has the correct format with 'timestamp' and 'energy_kW' columns.")

# -----------------------------
# 🏢 PAGE 5: Buildings
# -----------------------------
elif page == "🏢 Buildings":
    st.header("🏢 Building Management")
    st.markdown("Manage your building portfolio and monitor energy performance")

    # Quick stats
    total_buildings = len(st.session_state.buildings)
    total_sq_ft = sum([int(b.get("Size", "0").replace(",", "").split()[0]) for b in st.session_state.buildings if "Size" in b])
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Buildings", total_buildings)
    with col2:
        st.metric("Total Area", f"{total_sq_ft:,} sq ft")
    with col3:
        st.metric("Avg Efficiency", "87%")
    with col4:
        st.metric("Monthly Savings", "$15,340")

    # Buildings table with enhanced display
    st.markdown("### 📊 Building Portfolio")
    
    if st.session_state.buildings:
        df_buildings = pd.DataFrame(st.session_state.buildings)
        
        # Add some calculated metrics
        df_display = df_buildings.copy()
        df_display['Energy_Efficiency'] = [f"{np.random.randint(82, 95)}%" for _ in range(len(df_display))]
        df_display['Monthly_Cost'] = [f"${np.random.randint(2000, 8000):,}" for _ in range(len(df_display))]
        df_display['Status'] = ['🟢 Active' for _ in range(len(df_display))]
        
        # Reorder columns for better display
        display_columns = ['Name', 'Type', 'Location', 'Size', 'Energy_Efficiency', 'Monthly_Cost', 'Status', 'Added']
        df_display = df_display[display_columns]
        df_display.columns = ['Building Name', 'Type', 'Location', 'Size', 'Efficiency', 'Monthly Cost', 'Status', 'Date Added']
        
        st.dataframe(df_display, use_container_width=True, hide_index=True)
        
        # Building selection for detailed view
        selected_building = st.selectbox("Select Building for Details", 
                                       [b["Name"] for b in st.session_state.buildings])
        
        if selected_building:
            building_data = next(b for b in st.session_state.buildings if b["Name"] == selected_building)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown(f"### 🏢 {selected_building}")
                st.write(f"**Type:** {building_data['Type']}")
                st.write(f"**Location:** {building_data['Location']}")
                st.write(f"**Size:** {building_data.get('Size', 'N/A')}")
                st.write(f"**Added:** {building_data['Added']}")
            
            with col2:
                # Generate sample energy chart for selected building
                dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
                base_consumption = {"Commercial": 80, "Residential": 45, "Retail": 120, "Industrial": 200}
                base = base_consumption.get(building_data['Type'], 60)
                
                energy_data = base + 20 * np.sin(np.arange(30) / 30 * 2 * np.pi * 3) + np.random.normal(0, 8, 30)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=dates, y=energy_data,
                    mode='lines+markers',
                    name='Daily Energy',
                    line=dict(color='#1E90FF', width=2),
                    marker=dict(size=4)
                ))
                fig.update_layout(
                    title=f"{selected_building} - 30 Day Energy Profile",
                    xaxis_title="Date",
                    yaxis_title="Energy (kWh)",
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)

    # Add new building section
    st.markdown("### ➕ Add New Building")
    
    # Quick add mode (from home page)
    if st.session_state.get("quick_add", False):
        st.info("🚀 Quick Add Mode Activated!")
        st.session_state.quick_add = False
    
    with st.expander("Add New Building", expanded=st.session_state.get("quick_add", False)):
        with st.form("add_building"):
            col1, col2 = st.columns(2)
            
            with col1:
                name = st.text_input("Building Name*", placeholder="e.g., Office Tower D")
                btype = st.selectbox("Building Type*", ["Commercial", "Residential", "Retail", "Industrial", "Mixed Use"])
                location = st.text_input("Location*", placeholder="e.g., New York, NY")
            
            with col2:
                size = st.text_input("Size", placeholder="e.g., 45,000 sq ft")
                floors = st.number_input("Number of Floors", min_value=1, max_value=200, value=10)
                year_built = st.number_input("Year Built", min_value=1900, max_value=2024, value=2015)
            
            submitted = st.form_submit_button("🏢 Add Building", use_container_width=True)
            
            if submitted:
                if name and btype and location:
                    new_building = {
                        "Name": name,
                        "Type": btype,
                        "Location": location,
                        "Size": size if size else "N/A",
                        "Floors": floors,
                        "Year_Built": year_built,
                        "Added": datetime.now().strftime("%Y-%m-%d")
                    }
                    st.session_state.buildings.append(new_building)
                    st.success(f"✅ Successfully added {name} to your portfolio!")
                    st.balloons()
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("❌ Please fill in all required fields (marked with *)")

    # Bulk operations
    st.markdown("### 🔧 Bulk Operations")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Generate Portfolio Report"):
            st.info("📄 Portfolio report generated! (Feature simulated)")
    
    with col2:
        if st.button("📥 Export Buildings CSV"):
            csv = pd.DataFrame(st.session_state.buildings).to_csv(index=False)
            st.download_button(
                "Download CSV",
                data=csv,
                file_name=f"buildings_portfolio_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    with col3:
        if st.button("🔄 Refresh All Data"):
            st.success("✅ All building data refreshed!")

# -----------------------------
# 💡 PAGE 6: Insights
# -----------------------------
elif page == "💡 Insights":
    st.header("💡 Energy Insights & Recommendations")
    st.markdown("AI-powered insights to optimize energy consumption and sustainability")

    # Select building for insights
    if st.session_state.buildings:
        selected_building = st.selectbox("Select Building for Analysis", 
                                       [b["Name"] for b in st.session_state.buildings])
        
        building_info = next(b for b in st.session_state.buildings if b["Name"] == selected_building)
        building_type = building_info["Type"]
    else:
        st.warning("No buildings found. Please add buildings in the Buildings section first.")
        st.stop()

    # Generate insights based on building type
    st.markdown("### 🌱 Sustainability Analysis")
    
    # Renewable Readiness Score
    readiness_score = np.random.uniform(0.75, 0.95)
    st.markdown(f"### 📈 **Renewable Readiness Score: {readiness_score:.1f}/10**")
    st.progress(readiness_score / 10)

    # Key insights
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"""
        ### 🎯 **Personalized Recommendations for {selected_building}**
        
        **🔋 Energy Storage**
        - Install 75kWh battery system to shift peak loads (6-8 PM)
        - Estimated savings: $2,340/month
        - Payback period: 4.2 years
        
        **☀️ Solar Integration**  
        - Rooftop solar potential: 150kW capacity
        - Could offset 65% of daytime consumption
        - Annual CO₂ reduction: 45 tons
        
        **🌬️ HVAC Optimization**
        - Smart thermostat programming could save 18% monthly
        - Night setback: 65°F → 62°F (Oct-Mar)
        - Occupancy-based zones during low-traffic hours
        
        **📉 Peak Load Management**
        - Current peak: 145kW at 6:30 PM
        - Target reduction: 25kW (17% decrease)
        - Load shifting opportunities: EV charging, water heating
        
        **🏭 Equipment Upgrades**
        - LED lighting conversion: 8% energy reduction
        - High-efficiency motors: 12% savings on mechanical systems
        - Smart building automation: 15% overall optimization
        """)
    
    with col2:
        # Energy breakdown chart
        labels = ['HVAC', 'Lighting', 'Equipment', 'Other']
        if building_type == "Commercial":
            values = [45, 25, 20, 10]
        elif building_type == "Residential":
            values = [40, 15, 30, 15]
        elif building_type == "Retail":
            values = [35, 35, 20, 10]
        else:  # Industrial
            values = [25, 15, 50, 10]
        
        fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.4)])
        fig.update_layout(
            title=f"Energy Breakdown - {building_type}",
            height=300,
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Savings potential
        st.markdown("### 💰 Savings Potential")
        monthly_savings = np.random.randint(1500, 4000)
        annual_savings = monthly_savings * 12
        co2_reduction = np.random.randint(25, 60)
        
        st.metric("Monthly Savings", f"${monthly_savings:,}", f"+${monthly_savings//4}")
        st.metric("Annual Savings", f"${annual_savings:,}")
        st.metric("CO₂ Reduction", f"{co2_reduction} tons/year")

    # Advanced analytics
    st.markdown("### 📊 Advanced Analytics")
    
    tab1, tab2, tab3 = st.tabs(["🔮 Forecasting", "📈 Trends", "⚡ Optimization"])
    
    with tab1:
        st.markdown("#### Energy Demand Forecast - Next 7 Days")
        
        # Generate 7-day forecast
        dates = pd.date_range(start=datetime.now(), periods=7*24, freq='H')
        base_demand = 80 if building_type == "Commercial" else 45
        hourly_pattern = 25 * np.sin((dates.hour - 8) / 24 * 2 * np.pi)
        daily_variation = 10 * np.sin(dates.dayofyear / 365 * 2 * np.pi)
        forecast_demand = base_demand + hourly_pattern + daily_variation + np.random.normal(0, 5, len(dates))
        forecast_demand = np.clip(forecast_demand, 10, 150)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates, y=forecast_demand,
            mode='lines', name='Predicted Demand',
            line=dict(color='#FF6B35', width=2)
        ))
        
        # Add peak hours highlighting
        peak_hours = dates[dates.hour.isin([17, 18, 19])]
        peak_demand = forecast_demand[dates.hour.isin([17, 18, 19])]
        fig.add_trace(go.Scatter(
            x=peak_hours, y=peak_demand,
            mode='markers', name='Peak Hours',
            marker=dict(color='red', size=6)
        ))
        
        fig.update_layout(
            title="7-Day Energy Demand Forecast",
            xaxis_title="Time",
            yaxis_title="Energy Demand (kW)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("#### Monthly Energy Trends")
        
        # Generate monthly trend data
        months = pd.date_range(start='2023-01-01', periods=12, freq='M')
        seasonal_effect = 20 * np.sin((months.month - 6) / 12 * 2 * np.pi)
        base_monthly = 35000  # kWh
        monthly_consumption = base_monthly + seasonal_effect * 1000 + np.random.normal(0, 2000, 12)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=months, y=monthly_consumption,
            mode='lines+markers', name='Monthly Consumption',
            line=dict(color='#1E90FF', width=3),
            marker=dict(size=8)
        ))
        
        # Add trend line
        z = np.polyfit(range(12), monthly_consumption, 1)
        p = np.poly1d(z)
        fig.add_trace(go.Scatter(
            x=months, y=p(range(12)),
            mode='lines', name='Trend',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title="12-Month Energy Consumption Trend",
            xaxis_title="Month",
            yaxis_title="Energy Consumption (kWh)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Trend insights
        trend_slope = z[0]
        if trend_slope > 0:
            st.info(f"📈 Energy consumption is increasing by {abs(trend_slope):.0f} kWh/month")
        else:
            st.success(f"📉 Energy consumption is decreasing by {abs(trend_slope):.0f} kWh/month")
    
    with tab3:
        st.markdown("#### Optimization Opportunities")
        
        # Optimization recommendations with impact
        optimizations = [
            {"Action": "Install Smart Thermostats", "Savings": "18%", "Cost": "$2,500", "Payback": "8 months"},
            {"Action": "LED Lighting Conversion", "Savings": "8%", "Cost": "$5,000", "Payback": "14 months"},
            {"Action": "Peak Load Shifting", "Savings": "12%", "Cost": "$1,200", "Payback": "6 months"},
            {"Action": "Solar Panel Installation", "Savings": "35%", "Cost": "$25,000", "Payback": "5.2 years"},
            {"Action": "Energy Management System", "Savings": "15%", "Cost": "$8,000", "Payback": "18 months"}
        ]
        
        df_opt = pd.DataFrame(optimizations)
        st.dataframe(df_opt, use_container_width=True, hide_index=True)
        
        # ROI visualization
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[float(opt["Savings"].rstrip('%')) for opt in optimizations],
            y=[float(opt["Cost"].replace('$', '').replace(',', '')) for opt in optimizations],
            text=[opt["Action"] for opt in optimizations],
            textposition="top center",
            marker=dict(size=12, color='#1E90FF'),
            name='Optimization Options'
        ))
        
        fig.update_layout(
            title="Optimization Options: Savings vs Investment",
            xaxis_title="Energy Savings (%)",
            yaxis_title="Investment Cost ($)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    # Report generation
    st.markdown("### 📄 Generate Reports")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Energy Audit Report"):
            with st.spinner("Generating comprehensive energy audit..."):
                time.sleep(2)
                st.success("✅ Energy Audit Report generated!")
                
                report_data = f"""# Energy Audit Report - {selected_building}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Building Information
- Name: {selected_building}
- Type: {building_type}
- Renewable Readiness Score: {readiness_score:.1f}/10

## Key Recommendations
1. Install 75kWh battery system - $2,340/month savings
2. 150kW rooftop solar - 65% offset potential
3. Smart HVAC optimization - 18% savings
4. Peak load management - 17% reduction target

## Annual Impact
- Cost Savings: ${annual_savings:,}
- CO₂ Reduction: {co2_reduction} tons
- Energy Efficiency Improvement: 25%
"""
                
                st.download_button(
                    "📥 Download Audit Report",
                    data=report_data,
                    file_name=f"energy_audit_{selected_building.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.md",
                    mime="text/markdown"
                )
    
    with col2:
        if st.button("🌱 Sustainability Report"):
            st.info("🌱 Sustainability report with carbon footprint analysis ready!")
    
    with col3:
        if st.button("💰 ROI Analysis"):
            st.info("💰 Return on Investment analysis for all recommendations prepared!")

    # Final summary
    st.markdown("---")
    st.markdown("### 🎯 Executive Summary")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        **🏢 Building Profile**
        - Type: {building_type}
        - Readiness Score: {readiness_score:.1f}/10
        - Optimization Potential: High
        """)
    
    with col2:
        st.markdown(f"""
        **💰 Financial Impact**  
        - Monthly Savings: ${monthly_savings:,}
        - Annual Savings: ${annual_savings:,}
        - 5-Year Value: ${annual_savings * 5:,}
        """)
    
    with col3:
        st.markdown(f"""
        **🌍 Environmental Impact**
        - CO₂ Reduction: {co2_reduction} tons/year
        - Renewable Potential: 65%
        - Sustainability Grade: A-
        """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #666; padding: 20px;">
        <p>⚡ <strong>EnergySense</strong> - AI for Sustainable Buildings | 
        Built with Streamlit & Time Series Foundation Models</p>
        <p>🌱 Supporting global decarbonization through intelligent energy management</p>
    </div>
    """,
    unsafe_allow_html=True
)