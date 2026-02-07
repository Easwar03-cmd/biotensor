import numpy as np
from biotensor.decoder import ReadoutLayer
from biotensor.reservoir import BioReservoir
from biotensor.encoder import SpikeEncoder
from biotensor.audio import AudioProcessor
from brian2 import *
import sys
import os
sys.path.append(os.getcwd())


# CONFIG
seed_val = 42
N_NEURONS = 1000
DURATION = 200*ms

# --- 1. GENERATE TONES ---


def generate_tone(freq, sr=16000):
    t = np.linspace(0, 1.0, sr)
    return 0.5 * np.sin(2 * np.pi * freq * t)


print("--- HEARING TEST START ---")
processor = AudioProcessor(n_mfcc=20)

# Create Dataset: 20 Low Beeps vs 20 High Beeps
X_raw = []
y = []

for _ in range(20):
    X_raw.append(generate_tone(300))  # Low
    y.append(0)
    X_raw.append(generate_tone(2000))  # High
    y.append(1)

# Convert to MFCC
print("Encoding Sound into Biological Signals...")
X_mfcc = [processor.process_file_from_array(a) for a in X_raw]

# --- 2. THE ORGANOID LISTENS ---
# Use the OPTIMIZED parameters from yesterday
# "The Enlightened Monk" config
print("Initializing BioReservoir...")
encoder = SpikeEncoder(duration=DURATION, max_freq=100*Hz)
decoder = ReadoutLayer(regularization=1.0)
reservoir_states = []

for i, sound_pattern in enumerate(X_mfcc):
    seed(seed_val)
    np.random.seed(seed_val)
    start_scope()

    # Tuned "Monk" Parameters
    eqs = '''
    dv/dt = (I - v) / (50*ms) : 1 (unless refractory)
    dI/dt = -I / (5*ms) : 1 
    '''
    neurons = NeuronGroup(N_NEURONS, eqs,
                          threshold='v>1.7', reset='v=0',
                          refractory=15*ms, method='exact')
    neurons.v = 0
    synapses = Synapses(neurons, neurons, on_pre='I += 0.5')
    synapses.connect(p=0.1)
    mon = SpikeMonitor(neurons)
    net = Network(neurons, synapses, mon)

    # Connect Ear to Brain
    # Note: MFCC is 2D, but our encoder expects 1D. We flatten it.
    # We take the AVERAGE over time to create a spatial pattern (simplification for v1)
    sound_vector = np.mean(sound_pattern, axis=1)

    input_group = encoder.encode_image(
        sound_vector)  # We reuse the image encoder!
    input_syn = Synapses(input_group, neurons, on_pre='v += 0.5')
    input_syn.connect(p=0.3)
    net.add(input_group)
    net.add(input_syn)

    net.run(DURATION)
    reservoir_states.append(mon.count[:])

    if i % 10 == 0:
        print(f"Processing sound {i}...")

# --- 3. RESULTS ---
split = 30
decoder.train(reservoir_states[:split], y[:split])

print("\n--- FINAL HEARING EXAM ---")
correct = 0
tests = reservoir_states[split:]
labels = y[split:]

for i, state in enumerate(tests):
    pred = decoder.predict(state)[0]
    p_label = 0 if pred < 0.5 else 1
    target = labels[i]
    tone_type = "Low Hum" if target == 0 else "High Whistle"

    status = "✅ PASS" if p_label == target else "❌ FAIL"
    print(f"Sound: {tone_type} | Organoid prediction: {pred:.3f} | {status}")
    if p_label == target:
        correct += 1

print(f"\nAuditory Accuracy: {(correct/len(tests))*100}%")
