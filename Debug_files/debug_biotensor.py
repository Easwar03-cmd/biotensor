import numpy as np
import matplotlib.pyplot as plt
from biotensor.reservoir import BioReservoir
from biotensor.vision import VisionLoader
from brian2 import *
import sys
import os
sys.path.append(os.getcwd())


# --- SETUP ---
start_scope()
print("--- DIAGNOSTIC MODE ---")

# 1. Inspect Data
loader = VisionLoader()
X, y = loader.get_pair(digit_A=0, digit_B=1, n_samples=1)
image = X[0]
print(f"Data Check: Image Shape={image.shape}, Max Value={np.max(image)}")
if np.max(image) == 0:
    print("CRITICAL ERROR: The image is blank (all zeros). VisionLoader is failing.")
    sys.exit()

# 2. Inspect Encoder (Manually)
print("Building Encoder...")
max_freq = 100*Hz
# Create the input group manually to verify it exists
input_group = PoissonGroup(len(image), rates=image * max_freq)

# 3. Inspect Brain
print("Building Brain...")
organoid = BioReservoir(n_neurons=500, connection_prob=0.1)

# 4. Connect
print("Connecting...")
# We do this manually to ensure no 'Magic Network' issues
organoid.net.add(input_group)
syn = Synapses(input_group, organoid.neurons,
               on_pre='v += 2.0')  # Massive shock
syn.connect(p=0.5)
organoid.net.add(syn)

# 5. Monitor EVERYTHING
# We monitor the INPUT (to see if it enters) and the BRAIN (to see if it reacts)
mon_input = SpikeMonitor(input_group)
organoid.net.add(mon_input)

# 6. Run
print("Running Simulation (100ms)...")
organoid.net.run(100*ms)

# 7. REPORT CARD
spikes_in = mon_input.num_spikes
spikes_brain = organoid.spike_monitor.num_spikes

print(f"\n--- DIAGNOSTIC RESULTS ---")
print(f"INPUT SPIKES: {spikes_in}")
print(f"BRAIN SPIKES: {spikes_brain}")

if spikes_in == 0:
    print("FAIL: The encoder is not generating any electricity.")
elif spikes_brain == 0:
    print("FAIL: The input is firing, but the brain is ignoring it.")
else:
    print("SUCCESS: The system is alive.")

# 8. VISUALIZE
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(mon_input.t/ms, mon_input.i, '.k')
plt.title(f"Input Activity ({spikes_in} spikes)")
plt.xlabel("Time (ms)")
plt.ylabel("Input Pixel ID")

plt.subplot(1, 2, 2)
plt.plot(organoid.spike_monitor.t/ms, organoid.spike_monitor.i, '.r')
plt.title(f"Brain Activity ({spikes_brain} spikes)")
plt.xlabel("Time (ms)")
plt.ylabel("Neuron ID")

plt.show()
