import sys
import os

# --- 1. FORCE THE PATH (The "Nuclear" Fix) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

print(f"DEBUG: Project Root added to path: {project_root}")
# ---------------------------------------------------------

import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from biotensor.encoder import SpikeEncoder  # noqa: E402
from biotensor.audio import AudioProcessor  # noqa: E402
from biotensor.speech import VowelSynthesizer  # noqa: E402
from brian2 import *  # noqa: E402


# CONFIG
seed_val = 100
N_NEURONS = 1000
DURATION = 200*ms

print("--- GENERATING SPEECH VISUALIZATION ---")

# 1. Generate ONE 'Ahh' and ONE 'Eee'
synth = VowelSynthesizer()
processor = AudioProcessor(n_mfcc=13)

# Sound 1: AHH
raw_ahh = synth.generate_vowel('a')
mfcc_ahh = processor.process_file_from_array(raw_ahh)
vec_ahh = mfcc_ahh[:, mfcc_ahh.shape[1] // 2]  # Center slice

# Sound 2: EEE
raw_eee = synth.generate_vowel('i')
mfcc_eee = processor.process_file_from_array(raw_eee)
vec_eee = mfcc_eee[:, mfcc_eee.shape[1] // 2]  # Center slice

encoder = SpikeEncoder(duration=DURATION, max_freq=120*Hz)

# --- RUN FOR 'AHH' ---
print("Simulating Brain response to 'AHH'...")
seed(seed_val)
np.random.seed(seed_val)
start_scope()

# "Enlightened Monk" Params
eqs = '''
dv/dt = (I - v) / (50*ms) : 1 (unless refractory)
dI/dt = -I / (5*ms) : 1 
'''
neurons = NeuronGroup(N_NEURONS, eqs, threshold='v>1.7',
                      reset='v=0', refractory=15*ms, method='exact')
neurons.v = 0
synapses = Synapses(neurons, neurons, on_pre='I += 0.5')
synapses.connect(p=0.1)
mon_ahh = SpikeMonitor(neurons)
net = Network(neurons, synapses, mon_ahh)

input_group = encoder.encode_image(vec_ahh)
input_syn = Synapses(input_group, neurons, on_pre='v += 0.6')
input_syn.connect(p=0.4)
net.add(input_group)
net.add(input_syn)

net.run(DURATION)
times_ahh = mon_ahh.t/ms
indices_ahh = mon_ahh.i

# --- RUN FOR 'EEE' ---
print("Simulating Brain response to 'EEE'...")
seed(seed_val)
np.random.seed(seed_val)
start_scope()

# Rebuild exact same brain
neurons = NeuronGroup(N_NEURONS, eqs, threshold='v>1.7',
                      reset='v=0', refractory=15*ms, method='exact')
neurons.v = 0
synapses = Synapses(neurons, neurons, on_pre='I += 0.5')
synapses.connect(p=0.1)
mon_eee = SpikeMonitor(neurons)
net = Network(neurons, synapses, mon_eee)

input_group = encoder.encode_image(vec_eee)
input_syn = Synapses(input_group, neurons, on_pre='v += 0.6')
input_syn.connect(p=0.4)
net.add(input_group)
net.add(input_syn)

net.run(DURATION)
times_eee = mon_eee.t/ms
indices_eee = mon_eee.i

# --- PLOT ---
print("Plotting Figure 2...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

# Plot AHH (Blue)
ax1.plot(times_ahh, indices_ahh, '.b', markersize=2, alpha=0.6)
ax1.set_title(f"Auditory Response: 'AHH' /a/\n(Spikes: {len(indices_ahh)})")
ax1.set_xlabel("Time (ms)")
ax1.set_ylabel("Neuron Index")
ax1.set_ylim(0, N_NEURONS)
ax1.grid(True, alpha=0.2)

# Plot EEE (Red)
ax2.plot(times_eee, indices_eee, '.r', markersize=2, alpha=0.6)
ax2.set_title(f"Auditory Response: 'EEE' /i/\n(Spikes: {len(indices_eee)})")
ax2.set_xlabel("Time (ms)")
ax2.grid(True, alpha=0.2)

plt.suptitle(
    f"BioTensor Phoneme Discrimination\n(Seed {seed_val})", fontsize=14)
plt.tight_layout()
plt.show()
