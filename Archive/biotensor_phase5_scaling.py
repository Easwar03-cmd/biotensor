import numpy as np
from biotensor.decoder import ReadoutLayer
from biotensor.audio import AudioProcessor
from biotensor.speech import VowelSynthesizer
from biotensor.vision import VisionLoader
from brian2 import *
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

prefs.codegen.target = 'numpy'  # Safe Mode


# --- CONFIGURATION ---
seed_val = 2060
N_NEURONS = 1000
DURATION = 100*ms
N_SAMPLES = 100

print("=========================================")
print("   BIOTENSOR PHASE 5: WEIGHT SCALING     ")
print("=========================================")
print("Objective: 100% Accuracy.")
print("Strategy: Eraser Learning + Post-Training Normalization.")

# 1. LOAD DATA
loader = VisionLoader()
X_img, y_img = loader.get_pair(digit_A=0, digit_B=1, n_samples=N_SAMPLES)

synth = VowelSynthesizer()
processor = AudioProcessor(n_mfcc=13)
X_audio_raw = []
for label in y_img:
    if label == 0:
        raw_sound = synth.generate_vowel('a')
    else:
        raw_sound = synth.generate_vowel('i')
    mfcc = processor.process_file_from_array(raw_sound)
    center_slice = mfcc[:, mfcc.shape[1] // 2]
    X_audio_raw.append(center_slice)

# Constants
VIS_END = len(X_img[0].flatten())
TOTAL_CHANNELS = VIS_END + len(X_audio_raw[0])
TAU_MAX = 50*ms

# --- PHASE A: LEARNING (The Eraser Method) ---
print("\n[A] Phase A: The Learning Process...")

eqs = '''
dv/dt = (I - v + I_teach) / (20*ms) : 1 (unless refractory)
dI/dt = -I / (5*ms) : 1 
I_teach : 1 
'''

print("    > Initializing Synaptic Matrix...")
Synaptic_Weights = np.random.uniform(0.2, 0.5, (TOTAL_CHANNELS, N_NEURONS))

# LEARNING LOOP (Standard Eraser Logic)
for idx in range(80):
    # 1. PREPARE DATA
    img_vec = X_img[idx].flatten()
    img_norm = img_vec / (np.max(img_vec)+1e-8)
    aud_vec = X_audio_raw[idx]
    aud_norm = (aud_vec-np.min(aud_vec))/(np.max(aud_vec)-np.min(aud_vec)+1e-8)

    indices = []
    times = []
    for i, val in enumerate(img_norm):
        if val > 0.1:
            indices.append(i)
            times.append((1.0-val)*TAU_MAX)
    for i, val in enumerate(aud_norm):
        if val > 0.1:
            indices.append(VIS_END + i)
            times.append((1.0-val)*TAU_MAX)

    if len(indices) == 0:
        indices = [0]
        times = [0]*ms
    else:
        ord = np.argsort(times)
        indices = np.array(indices)[ord]
        times = np.array(times)[ord]*second

    # 2. RUN SIMULATION
    start_scope()
    neurons = NeuronGroup(N_NEURONS, eqs, threshold='v>1.5',
                          reset='v=0', refractory=3*ms, method='exact')
    neurons.v = 0

    # TEACHER SIGNAL
    if y_img[idx] == 0:
        neurons.I_teach[:500] = 1.5
        neurons.I_teach[500:] = -1.0
    else:
        neurons.I_teach[:500] = -1.0
        neurons.I_teach[500:] = 1.5

    input_group = SpikeGeneratorGroup(TOTAL_CHANNELS, indices, times)
    mon_in = SpikeMonitor(input_group)
    mon_out = SpikeMonitor(neurons)

    syn = Synapses(input_group, neurons, 'w : 1', on_pre='v += w')
    syn.connect()
    syn.w = Synaptic_Weights.flatten()

    net = Network(neurons, input_group, syn, mon_in, mon_out)
    net.run(DURATION)

    # 3. APPLY PLASTICITY (Eraser Rule)
    spikes_in = mon_in.i[:]
    spikes_out = mon_out.i[:]
    unique_in = np.unique(spikes_in)
    unique_out = np.unique(spikes_out)

    # Reward Correct
    Synaptic_Weights[np.ix_(unique_in, unique_out)] += 0.2

    # Punish Wrong (The Eraser)
    if y_img[idx] == 0:
        wrong_neurons = np.arange(500, 1000)
    else:
        wrong_neurons = np.arange(0, 500)

    Synaptic_Weights[np.ix_(unique_in, wrong_neurons)] -= 0.1

    # Clamp
    Synaptic_Weights = np.clip(Synaptic_Weights, 0.0, 10.0)

    if idx % 20 == 0:
        print(f"    > Learned sample {idx}...")

print(f"    > Training Complete.")

# --- THE FIX: POST-TRAINING SCALING ---
print("\n[FIX] Equalizing Neuron Strength...")
# We calculate the total synaptic weight coming into each neuron
total_input_weight = np.sum(Synaptic_Weights, axis=0)  # Sum down columns
total_input_weight[total_input_weight == 0] = 1.0

# Target Weight Sum (e.g., 50.0)
# We force every neuron to have the SAME total receiving capacity
TARGET_CAPACITY = 50.0
scaling_factors = TARGET_CAPACITY / total_input_weight

# Apply scaling
# This makes the "One" neurons (which have small weights) STRONGER
# And keeps "Zero" neurons (which have large weights) normal
Synaptic_Weights = Synaptic_Weights * scaling_factors[np.newaxis, :]
print("    > Weights Scaled. David and Goliath are now equal.")


# --- PHASE B: TESTING (Recall) ---
print("\n[B] Phase B: Recall Test (Audio DELETED)...")

decoder = ReadoutLayer(regularization=1.0)
reservoir_states = []

for idx in range(len(X_img)):
    start_scope()

    img_vec = X_img[idx].flatten()
    img_norm = img_vec / (np.max(img_vec)+1e-8)
    indices = []
    times = []
    for i, val in enumerate(img_norm):
        if val > 0.1:
            indices.append(i)
            times.append((1.0-val)*TAU_MAX)

    if len(indices) == 0:
        indices = [0]
        times = [0]*ms
    else:
        ord = np.argsort(times)
        indices = np.array(indices)[ord]
        times = np.array(times)[ord]*second

    neurons = NeuronGroup(N_NEURONS, eqs, threshold='v>1.5',
                          reset='v=0', refractory=3*ms, method='exact')
    neurons.v = 0
    neurons.I_teach = 0

    input_group = SpikeGeneratorGroup(TOTAL_CHANNELS, indices, times)

    syn = Synapses(input_group, neurons, 'w : 1', on_pre='v += w')
    syn.connect()
    syn.w = Synaptic_Weights.flatten()

    # LATERAL INHIBITION (Winner-Take-All)
    syn_inhib = Synapses(neurons, neurons, 'w_inhib : 1',
                         on_pre='v -= w_inhib')
    syn_inhib.connect(
        condition='(i < 500 and j >= 500) or (i >= 500 and j < 500)', p=0.05)
    syn_inhib.w_inhib = 2.0

    mon = SpikeMonitor(neurons)
    net = Network(neurons, input_group, syn, syn_inhib, mon)
    net.run(DURATION)

    reservoir_states.append(mon.count[:])

    if idx % 20 == 0:
        print(f"    > Tested memory recall on sample {idx}...")

# VERIFICATION
split = int(len(X_img) * 0.8)
decoder.train(reservoir_states[:split], y_img[:split])

correct = 0
tests = reservoir_states[split:]
labels = y_img[split:]

print("\n--- RESULTS: SELF-ORGANIZED MEMORY ---")
for i, state in enumerate(tests):
    pred = decoder.predict(state)[0]
    p_label = 0 if pred < 0.5 else 1
    target = labels[i]

    concept = "Zero" if target == 0 else "One"
    status = "✅ RECALLED" if p_label == target else "❌ FAILED"

    print(f"Vision: {concept} | Brain Output: {pred:.3f} | {status}")
    if p_label == target:
        correct += 1

acc = (correct / len(tests)) * 100
print(f"\nSTDP ACCURACY: {acc:.1f}%")

if acc >= 90:
    print("🏆 GRAND SUCCESS: The brain taught itself!")
