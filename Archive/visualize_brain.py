import numpy as np
import matplotlib.pyplot as plt
from biotensor.encoder import SpikeEncoder
from biotensor.reservoir import BioReservoir
from biotensor.vision import VisionLoader
from brian2 import *
import sys
import os
sys.path.append(os.getcwd())


# CONFIG
seed_value = 42
N_NEURONS = 1000
DURATION = 200*ms

print("--- GENERATING RESEARCH VISUALIZATION ---")

# 1. Load ONE '0' and ONE '1'
loader = VisionLoader()
X, y = loader.get_pair(digit_A=0, digit_B=1, n_samples=2)

img_zero = X[0]  # The Digit 0
img_one = X[1]  # The Digit 1

encoder = SpikeEncoder(duration=DURATION, max_freq=100*Hz)

# --- RUN FOR DIGIT 0 ---
print("Simulating Brain response to Digit '0'...")
seed(seed_value)
np.random.seed(seed_value)
brain_zero = BioReservoir(n_neurons=N_NEURONS)

spikes_zero = encoder.encode_image(img_zero)
# <--- Synapse is created here automatically
brain_zero.connect_input(spikes_zero)
brain_zero.run_simulation(DURATION)

# Extract Data
times_0 = brain_zero.spike_monitor.t/ms
indices_0 = brain_zero.spike_monitor.i

# --- RUN FOR DIGIT 1 ---
print("Simulating Brain response to Digit '1'...")
seed(seed_value)
np.random.seed(seed_value)  # SAME SEED (Important!)
brain_one = BioReservoir(n_neurons=N_NEURONS)

spikes_one = encoder.encode_image(img_one)
brain_one.connect_input(spikes_one)
brain_one.run_simulation(DURATION)

# Extract Data
times_1 = brain_one.spike_monitor.t/ms
indices_1 = brain_one.spike_monitor.i

# --- PLOT THE DIFFERENCE ---
print("Plotting Figure 1...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

# Plot 0 (Blue)
ax1.plot(times_0, indices_0, '.b', markersize=2, alpha=0.6)
ax1.set_title(f"Brain Activity: Digit '0'\n(Spikes: {len(indices_0)})")
ax1.set_xlabel("Time (ms)")
ax1.set_ylabel("Neuron Index")
ax1.set_ylim(0, N_NEURONS)
ax1.grid(True, alpha=0.2)

# Plot 1 (Red)
ax2.plot(times_1, indices_1, '.r', markersize=2, alpha=0.6)
ax2.set_title(f"Brain Activity: Digit '1'\n(Spikes: {len(indices_1)})")
ax2.set_xlabel("Time (ms)")
ax2.grid(True, alpha=0.2)

plt.suptitle(
    f"BioTensor Discrimination Analysis\n(Seed {seed_value})", fontsize=14)
plt.tight_layout()
plt.show()
