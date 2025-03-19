import streamlit as st

# ===================== Ensure Page Configuration is First =====================
if "config_set" not in st.session_state:
    st.set_page_config(
        layout="wide",
        page_title="AI-Powered Gravitational Wave & Quantum AI Navigation",
        page_icon="🌌"
    )
    st.session_state["config_set"] = True  # Prevents multiple calls to set_page_config

# Now import the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import requests
import pandas as pd
import time
from sklearn.metrics import mean_squared_error
from scipy.stats import norm
from scipy.integrate import solve_ivp

# ===================== Quantum Mechanics Constants =====================
hbar = 1.0  # Reduced Planck’s constant
m = 1.0  # Mass
dt = 0.01  # Time step
gamma_i = 0.05  # Damping Coefficient
alpha = 1.2  # Alpha Parameter

# ===================== Quantum Potential and Wavefunction =====================
def quantum_potential(x):
    return alpha * np.sin(x)**2  # Sample quantum potential function

def schrodinger_rhs(t, psi, x_grid):
    kinetic = -0.5 * hbar * np.gradient(np.gradient(psi, x_grid), x_grid) / m
    potential = quantum_potential(x_grid) * psi
    return -1j / hbar * (kinetic + potential)

def solve_schrodinger():
    x_grid = np.linspace(-5, 5, 200)
    psi_init = np.exp(-x_grid**2) * np.exp(1j * x_grid)
    sol = solve_ivp(lambda t, y: schrodinger_rhs(t, y, x_grid), [0, 2], psi_init, t_eval=np.linspace(0, 2, 100))
    return x_grid, sol.y

# Solve and visualize quantum wavefunction evolution
x_grid, psi_solutions = solve_schrodinger()

# ===================== Fix: Ensure Only One Graph Appears =====================
w_wavefunction_placeholder = st.empty()  # Create a container to hold the visualization

with w_wavefunction_placeholder.container():
    st.subheader("🔬 Quantum AI-Driven Wavefunction Evolution")

    # Clear previous figure before rendering a new one
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x_grid, np.abs(psi_solutions[:, -1])**2, label="Final Quantum State", color='magenta', linewidth=2)
    ax.set_xlabel("Position")
    ax.set_ylabel("Probability Density")
    ax.set_title("Quantum Wavefunction Evolution (AI-Driven Schrödinger Solution)")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    w_wavefunction_placeholder.pyplot(fig)  # Render the figure in the placeholder

# ===================== Auto-Refresh Every Few Seconds =====================
if "last_update" not in st.session_state:
    st.session_state.last_update = time.time()

time.sleep(5)
st.rerun()
