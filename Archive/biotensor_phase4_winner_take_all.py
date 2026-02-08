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
seed_val = 2040
N_NEURONS = 1000
DURATION = 100*ms
N_SAMPLES = 100

print("=========================================")
print("   BIOTENSOR PHASE 4: WINNER-TAKE-ALL    ")
print("=========================================")
print("Objective: Eliminate 'Fuzzy' Memories.")
print("Strategy: Lateral Inhibition (Competition).")

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

# --- PHASE A: EXPERIENCE ---
print("\n[A] Phase A: Training (Experience)...")
neuron_preferences = np.zeros((N_NEURONS, 2))

for idx in range(80):
    start_scope()
    seed(seed_val + idx)
    np.random.seed(seed_val + idx)

    # INPUT
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

    # BRAIN
    eqs = '''
    dv/dt = (I - v) / (30*ms) : 1 (unless refractory)
    dI/dt = -I / (5*ms) : 1 
    '''
    neurons = NeuronGroup(N_NEURONS, eqs, threshold='v>1.2',
                          reset='v=0', refractory=5*ms, method='exact')
    neurons.v = 0
    mon = SpikeMonitor(neurons)

    input_group = SpikeGeneratorGroup(TOTAL_CHANNELS, indices, times)
    syn_input = Synapses(input_group, neurons, on_pre='v += 1.5')
    syn_input.connect(p=0.1)

    net = Network(neurons, mon, input_group, syn_input)
    net.run(DURATION)

    spikes = mon.count[:]
    label = y_img[idx]
    neuron_preferences[:, label] += spikes

    if idx % 20 == 0:
        print(f"    > Learned from experience {idx}...")

# --- PHASE B: COMPETITIVE WIRING ---
print("\n[B] Phase B: Wiring the Competition...")

total_spikes = neuron_preferences[:, 0] + neuron_preferences[:, 1]
diff_spikes = neuron_preferences[:, 0] - neuron_preferences[:, 1]
loyalty_score = np.divide(diff_spikes, total_spikes + 1e-8)

sources = []
targets = []
weights = []

# Relaxed threshold to get robust teams
LOYALTY_THRESHOLD = 0.2

print(f"    > Creating Allies and Enemies (Threshold: {LOYALTY_THRESHOLD})...")
ally_count = 0
enemy_count = 0

for i in range(N_NEURONS):
    if abs(loyalty_score[i]) > LOYALTY_THRESHOLD and total_spikes[i] > 2:
        team_i = np.sign(loyalty_score[i])  # -1.0 (One) or +1.0 (Zero)

        for j in range(N_NEURONS):
            if i != j and abs(loyalty_score[j]) > LOYALTY_THRESHOLD:
                team_j = np.sign(loyalty_score[j])

                # LOGIC:
                # Same Team? -> SUPER BOOST (Positive)
                # Different Team? -> KILL (Negative)

                if team_i == team_j:
                    sources.append(i)
                    targets.append(j)
                    weights.append(2.0)  # Stronger Excitation
                    ally_count += 1
                else:
                    sources.append(i)
                    targets.append(j)
                    weights.append(-2.0)  # Strong Inhibition
                    enemy_count += 1

print(f"    > Wired {ally_count} Allies and {enemy_count} Enemies.")

# --- PHASE C: RECALL ---
print(f"\n[C] Phase C: Recall Test (Audio DELETED)...")

decoder = ReadoutLayer(regularization=1.0)
reservoir_states = []

for idx in range(len(X_img)):
    start_scope()
    seed(seed_val + idx)
    np.random.seed(seed_val + idx)

    # VISION ONLY
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

    # BRAIN WITH COMPETITION
    eqs = '''
    dv/dt = (I - v) / (30*ms) : 1 (unless refractory)
    dI/dt = -I / (5*ms) : 1 
    '''
    neurons = NeuronGroup(N_NEURONS, eqs, threshold='v>1.2',
                          reset='v=0', refractory=5*ms, method='exact')
    neurons.v = 0
    mon = SpikeMonitor(neurons)

    input_group = SpikeGeneratorGroup(TOTAL_CHANNELS, indices, times)
    syn_input = Synapses(input_group, neurons, on_pre='v += 1.5')
    syn_input.connect(p=0.1)

    network_objects = [neurons, mon, input_group, syn_input]

    # ADD COMPETITIVE SYNAPSES
    if len(sources) > 0:
        synapses = Synapses(neurons, neurons, 'w : 1', on_pre='I += w')
        # Explicit INT casting
        sources_arr = np.array(sources, dtype=int)
        targets_arr = np.array(targets, dtype=int)
        synapses.connect(i=sources_arr, j=targets_arr)
        synapses.w = weights
        network_objects.append(synapses)

    net = Network(network_objects)
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

print("\n--- RESULTS: GHOST IN THE MACHINE ---")
for i, state in enumerate(tests):
    pred = decoder.predict(state)[0]
    p_label = 0 if pred < 0.5 else 1
    target = labels[i]

    concept = "Zero" if target == 0 else "One"
    status = "✅ HALLUCINATED" if p_label == target else "❌ FAILED"

    print(
        f"Vision: {concept} | Audio: [SILENCE] | Brain Output: {pred:.3f} | {status}")
    if p_label == target:
        correct += 1

acc = (correct / len(tests)) * 100
print(f"\nFINAL ACCURACY: {acc:.1f}%")

if acc >= 95:
    print("🏆 MISSION ACCOMPLISHED: Perfect Associative Memory.")
