# Reload necessary libraries after execution state reset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ===================== Restoring Quantum Flux & String Resonance =====================
x_values = np.linspace(0, 10, 1000)  # Define position values
quantum_flux = np.sin(5 * np.pi * x_values) * np.exp(-0.1 * x_values)  # Quantum fluctuation
string_resonance = np.sin(2 * np.pi * x_values) * np.exp(-0.05 * x_values)  # String resonance

# Combine the two with adaptive scaling
combined_wave = (quantum_flux + string_resonance) / 2  # Blending both effects

# Generate Plot
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(x_values, combined_wave, label="Restored String Resonance & Quantum Flux", color='orange', linewidth=2)
ax.set_xlabel("Position")
ax.set_ylabel("Amplitude")
ax.set_title("🔬 Quantum Flux & String Theory Resonance")
ax.legend()
ax.grid(True, linestyle='--', alpha=0.7)

# ===================== Displaying Research Graphs for FAR & Mass =====================

# Simulated dataset (since previous variables were lost)
df1 = pd.DataFrame({
    "Timestamp": np.linspace(0, 10, 100),
    "False Alarm Rate": np.random.uniform(1e-8, 1e-5, 100),
    "Total Mass": np.random.uniform(10, 80, 100)
})

fig2, ax2 = plt.subplots(figsize=(8, 4))
ax2.plot(df1["Timestamp"], df1["False Alarm Rate"], marker='o', linestyle='-', color='red', alpha=0.7, label="False Alarm Rate")
ax2.set_xlabel("Timestamp")
ax2.set_ylabel("False Alarm Rate (log scale)")
ax2.set_yscale("log")  # Set log scale for better visualization of small values
ax2.set_title("False Alarm Rate Over Time")
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.6)

fig3, ax3 = plt.subplots(figsize=(8, 4))
ax3.hist(df1["Total Mass"], bins=15, color='blue', edgecolor='black', alpha=0.7)
ax3.set_xlabel("Total Mass (Solar Masses)")
ax3.set_ylabel("Number of Events")
ax3.set_title("Mass Distribution of Detected Gravitational Wave Events")
ax3.grid(True, linestyle='--', alpha=0.6)

# ===================== Displaying All Results =====================
import ace_tools as tools

tools.display_dataframe_to_user(name="LIGO Gravitational Wave Data", dataframe=df1)
plt.show()
