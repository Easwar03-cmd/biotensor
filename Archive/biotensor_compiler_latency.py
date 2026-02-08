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
seed_val = 9999
N_NEURONS = 1000
DURATION = 100*ms  # Latency coding is fast, we don't need 200ms
N_SAMPLES = 100

print("=========================================")
print("   BIOTENSOR OS: LATENCY COMPILER        ")
print("=========================================")
print("Mode: Deterministic Temporal Encoding")
print("Strategy: Converting Values into TIME (Precise Wavefronts)")

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

# 2. COMPILE TO LATENCY (The New Core Logic)
print("\n[2] Encoding Data as Time-Waves...")

# Constants
TAU_MAX = 50*ms  # The time window for the wave
INPUT_CHANNELS = len(X_img[0].flatten()) + len(X_audio_raw[0])
print(f" > Total Input Channels: {INPUT_CHANNELS}")

# We will pre-calculate the spike times for ALL samples
# But Brian2 requires us to rebuild the input group for each sample in a loop
# or use a single massive run. For simplicity, we stick to the loop but use SpikeGeneratorGroup.

# 3. EXECUTION
print("\n[3] Booting Organoid...")

# We use a simpler decoder now because the signal is cleaner
decoder = ReadoutLayer(regularization=1.0)
reservoir_states = []

for idx in range(len(X_img)):
    seed(seed_val + idx)
    np.random.seed(seed_val + idx)
    start_scope()

    # --- A. LATENCY ENCODER (CUSTOM) ---
    indices = []
    times = []

    # 1. Process Vision (Channels 0-63)
    img_vec = X_img[idx].flatten()
    img_norm = img_vec / (np.max(img_vec) + 1e-8)  # 0.0 to 1.0

    for i, val in enumerate(img_norm):
        if val > 0.1:  # Only spike if pixel is visible
            # KEY FORMULA: Higher Value = Earlier Time
            # 1.0 -> 0ms | 0.1 -> 50ms
            spike_time = (1.0 - val) * TAU_MAX
            indices.append(i)
            times.append(spike_time)

    # 2. Process Audio (Channels 64-75)
    aud_vec = X_audio_raw[idx]
    # Normalize Audio to 0.0-1.0
    aud_norm = (aud_vec - np.min(aud_vec)) / \
        (np.max(aud_vec) - np.min(aud_vec) + 1e-8)

    offset = len(img_norm)  # Start audio indices after vision
    for i, val in enumerate(aud_norm):
        if val > 0.1:
            spike_time = (1.0 - val) * TAU_MAX
            indices.append(offset + i)
            times.append(spike_time)

    # Sort times (Brian2 likes sorted spikes)
    # This creates the "Melody"
    sorted_order = np.argsort(times)
    indices = np.array(indices)[sorted_order]
    times = np.array(times)[sorted_order] * second  # Ensure units are correct

    # --- B. THE ORGANOID ---
    eqs = '''
    dv/dt = (I - v) / (30*ms) : 1 (unless refractory)
    dI/dt = -I / (5*ms) : 1 
    '''
    neurons = NeuronGroup(N_NEURONS, eqs,
                          threshold='v>1.2', reset='v=0',
                          refractory=5*ms, method='exact')
    neurons.v = 0

    # Recurrent Connections (The Echo Chamber)
    synapses = Synapses(neurons, neurons, on_pre='I += 0.5')
    synapses.connect(p=0.1)
    mon = SpikeMonitor(neurons)

    # --- C. INPUT INJECTION ---
    # We use SpikeGeneratorGroup instead of PoissonGroup
    # This plays the exact sequence we calculated above
    input_group = SpikeGeneratorGroup(INPUT_CHANNELS, indices, times)

    # Connect Inputs to Neurons
    syn_input = Synapses(input_group, neurons, on_pre='v += 0.8')
    syn_input.connect(p=0.1)

    net = Network(neurons, synapses, mon, input_group, syn_input)
    net.run(DURATION)

    reservoir_states.append(mon.count[:])

    if idx % 20 == 0:
        print(f" > Processing Batch {idx}...")

# 4. VERIFICATION
print("\n[4] Verifying Intelligence...")

split = int(len(X_img) * 0.8)
decoder.train(reservoir_states[:split], y_img[:split])

correct = 0
tests = reservoir_states[split:]
labels = y_img[split:]

print("\n--- DIAGNOSTIC RESULTS ---")
for i, state in enumerate(tests):
    pred = decoder.predict(state)[0]
    p_label = 0 if pred < 0.5 else 1
    target = labels[i]

    concept = "Zero (Ahh)" if target == 0 else "One (Eee)"
    status = "✅ PASS" if p_label == target else "❌ FAIL"

    print(f"Input: {concept} | Organoid Output: {pred:.3f} | {status}")
    if p_label == target:
        correct += 1

acc = (correct / len(tests)) * 100
print(f"\nSYSTEM ACCURACY: {acc:.1f}%")

if acc > 90:
    print("STATUS: OPERATIONAL. Latency Encoding successful.")
