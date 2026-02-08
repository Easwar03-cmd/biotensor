import numpy as np
from biotensor.decoder import ReadoutLayer
from biotensor.encoder import SpikeEncoder
from biotensor.audio import AudioProcessor
from biotensor.speech import VowelSynthesizer
from biotensor.vision import VisionLoader
from brian2 import *
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# --- CONFIGURATION ---
seed_val = 2024  # New stable seed
N_NEURONS = 1000
DURATION = 200*ms
N_SAMPLES = 60  # Increased from 20 to 60 for better learning

print("--- PHASE 3: MULTIMODAL ASSOCIATION (TUNED) ---")

# 1. PREPARE DATA
print("\n[1] Generating Sensory Data...")
loader = VisionLoader()
# Get more data to stabilize the learning
X_img, y_img = loader.get_pair(digit_A=0, digit_B=1, n_samples=N_SAMPLES)

synth = VowelSynthesizer()
processor = AudioProcessor(n_mfcc=13)
X_audio = []

for label in y_img:
    if label == 0:
        raw_sound = synth.generate_vowel('a')  # Ahh matches 0
    else:
        raw_sound = synth.generate_vowel('i')  # Eee matches 1

    mfcc = processor.process_file_from_array(raw_sound)
    # Use the center slice for stable vowel features
    center_slice = mfcc[:, mfcc.shape[1] // 2]
    X_audio.append(center_slice)

print(f"Data Ready: {len(X_img)} Pairs.")

# 2. INITIALIZE
print("\n[2] Building the Brain...")
encoder = SpikeEncoder(duration=DURATION, max_freq=120*Hz)
decoder = ReadoutLayer(regularization=2.0)  # Looser constraints
reservoir_states = []

# 3. RUN SIMULATION
for i in range(len(X_img)):
    seed(seed_val + i)
    np.random.seed(seed_val + i)
    start_scope()

    # "Enlightened Monk" Parameters
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

    # --- THE FIX: BALANCED INPUTS ---

    # Visual Input (Reduced weight)
    input_vision = encoder.encode_image(X_img[i])
    syn_vision = Synapses(input_vision, neurons, on_pre='v += 0.2')  # WAS 0.4
    syn_vision.connect(p=0.3)

    # Audio Input (Reduced weight)
    input_audio = encoder.encode_image(X_audio[i])
    syn_audio = Synapses(input_audio, neurons, on_pre='v += 0.2')  # WAS 0.4
    syn_audio.connect(p=0.3)

    net = Network(neurons, synapses, mon, input_vision,
                  syn_vision, input_audio, syn_audio)
    net.run(DURATION)

    reservoir_states.append(mon.count[:])
    if i % 10 == 0:
        print(f" > Learned Pair {i}...")

# 4. RESULTS
print("\n[3] Testing Associations...")
split = int(len(X_img) * 0.8)  # 80% Train
decoder.train(reservoir_states[:split], y_img[:split])

correct = 0
tests = reservoir_states[split:]
labels = y_img[split:]

print("\n--- MULTIMODAL REPORT CARD ---")
for i, state in enumerate(tests):
    pred = decoder.predict(state)[0]
    p_label = 0 if pred < 0.5 else 1
    target = labels[i]

    concept = "Zero+Ahh" if target == 0 else "One+Eee"
    status = "✅ MATCH" if p_label == target else "❌ MISMATCH"

    print(f"Concept: {concept} | Brain: {pred:.3f} | {status}")
    if p_label == target:
        correct += 1

acc = (correct / len(tests)) * 100
print(f"\nFinal Accuracy: {acc:.1f}%")
