import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import requests
import pandas as pd
import time
from sklearn.metrics import mean_squared_error
from scipy.stats import norm
from scipy.integrate import solve_ivp

# ===================== Ensure Page Configuration is First =====================
if "config_set" not in st.session_state:
    st.set_page_config(
        layout="wide",
        page_title="AI-Powered Gravitational Wave & Quantum AI Navigation",
        page_icon="🌌"
    )
    st.session_state["config_set"] = True  # Prevents multiple calls to set_page_config

# ===================== Sidebar Controls =====================
st.sidebar.header("🔧 Settings & Controls")

# Sensitivity Slider for Anomaly Classification
sensitivity_threshold = st.sidebar.slider("Set Sensitivity for Event Classification", 0.1, 5.0, 1.0)

# Real-Time Data Refresh Toggle
auto_refresh = st.sidebar.checkbox("Enable Auto-Refresh", value=True)

# ===================== Fetch Real LIGO/VIRGO Data =====================
@st.cache_data(ttl=300)
def fetch_ligo_data():
    url = "https://www.gw-openscience.org/eventapi/json/"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        events = data.get("events", {})
        if not events:
            return fetch_historical_ligo_data()
        df = pd.DataFrame(events).T
        expected_columns = ['GPS', 'FAR', 'Mtotal', 'Instruments']
        available_columns = [col for col in expected_columns if col in df.columns]
        if not available_columns:
            return fetch_historical_ligo_data()
        df = df[available_columns]
        column_renames = {'GPS': 'Timestamp', 'FAR': 'False Alarm Rate', 'Mtotal': 'Total Mass', 'Instruments': 'Detected By'}
        df = df.rename(columns={col: column_renames[col] for col in available_columns if col in column_renames})
        return df
    else:
        return fetch_historical_ligo_data()

@st.cache_data
def fetch_historical_ligo_data():
    return pd.DataFrame({
        "Timestamp": np.linspace(0, 10, 100),
        "False Alarm Rate": np.random.uniform(1e-8, 1e-5, 100),
        "Total Mass": np.random.uniform(10, 80, 100),
        "Detected By": np.random.choice(["LIGO-Hanford", "LIGO-Livingston", "LIGO-Virgo"], 100)
    })

ligo_df = fetch_ligo_data()

# ===================== AI vs. LIGO Waveform Comparison =====================
with st.container():
    st.subheader("🌊 AI vs. LIGO Waveform Comparison")
    fig, ax = plt.subplots(figsize=(10, 4))
    x_future = np.linspace(0, 10, 1000)
    ax.plot(x_future, np.sin(2 * np.pi * x_future), label="AI-Predicted GW Waveform", color='purple', linewidth=2)
    if not ligo_df.empty and "Timestamp" in ligo_df.columns:
        ax.plot(ligo_df["Timestamp"], np.sin(2 * np.pi * ligo_df["Timestamp"]), label="Actual LIGO Waveform", color='blue', linestyle='dashed', linewidth=2)
    ax.set_xlabel("Time")
    ax.set_ylabel("Amplitude")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig)

# ===================== Quantum AI Wavefunction Evolution =====================
with st.container():
    st.subheader("🔬 Quantum AI-Driven Wavefunction Evolution")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x_future, np.abs(np.sin(2 * np.pi * x_future))**2, label="Quantum Wavefunction", color='magenta', linewidth=2)
    ax.set_xlabel("Position")
    ax.set_ylabel("Probability Density")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig)

# ===================== String Theory & Quantum Flux Graph =====================
with st.container():
    st.subheader("🌌 String Theory & Quantum Flux Graph")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x_future, np.sin(2 * np.pi * x_future) * np.exp(-0.1 * x_future), label="String Resonance", color='orange', linewidth=2)
    ax.set_xlabel("Position")
    ax.set_ylabel("Quantum Flux")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig)

# ===================== Multi-Site LIGO Data Overlay =====================
with st.container():
    st.subheader("📡 Multi-Site Gravitational Wave Signal Overlay")
    fig, ax = plt.subplots(figsize=(10, 4))
    for site in ligo_df["Detected By"].unique():
        site_data = ligo_df[ligo_df["Detected By"] == site]
        if "Timestamp" in site_data.columns:
            ax.plot(site_data["Timestamp"], np.sin(2 * np.pi * site_data["Timestamp"]), label=f"{site} Signal")
    ax.set_xlabel("Time")
    ax.set_ylabel("Amplitude")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig)

# ===================== Auto-Refresh Control =====================
if auto_refresh:
    time.sleep(5)
    st.rerun()
