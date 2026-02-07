import numpy as np
from biotensor.decoder import ReadoutLayer
from biotensor.encoder import SpikeEncoder
from biotensor.speech import VowelSynthesizer
from biotensor.audio import AudioProcessor
from brian2 import *
import sys
import os
sys.path.append(os.getcwd())


# CONFIG
seed_val = 100  # New seed for speech
N_NEURONS = 1000
DURATION = 200*ms

print("--- BIO-SPEECH RECOGNITION ---")

# 1. Generate Data
synth = VowelSynthesizer()
processor = AudioProcessor(n_mfcc=13)  # Standard for speech
X_raw = []
y = []

print("Synthesizing 50 'Ahh's and 50 'Eee's...")
for _ in range(50):
    # Add slight random noise to pitch to make it realistic
    X_raw.append(synth.generate_vowel('a'))
    y.append(0)  # Label 0 = Ahh

    X_raw.append(synth.generate_vowel('i'))
    y.append(1)  # Label 1 = Eee

# Convert to MFCC (The "Ear")
X_mfcc = [processor.process_file_from_array(a) for a in X_raw]

# 2. Initialize Brain
# We use the "Enlightened Monk" parameters again
print("Initializing Auditory Cortex...")
encoder = SpikeEncoder(duration=DURATION, max_freq=120*Hz)
decoder = ReadoutLayer(regularization=2.0)
reservoir_states = []

# 3. Process Loop
for i, sound_pattern in enumerate(X_mfcc):
    seed(seed_val)
    np.random.seed(seed_val)
    start_scope()

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

    # SPEECH ENCODING TRICK:
    # Instead of averaging the whole sound, we take the "Center" of the vowel.
    # Vowels are stable in the middle.
    mid_point = sound_pattern.shape[1] // 2
    # Take the spectral slice at the center
    sound_vector = sound_pattern[:, mid_point]

    input_group = encoder.encode_image(sound_vector)
    # Stronger input for speech
    input_syn = Synapses(input_group, neurons, on_pre='v += 0.6')
    input_syn.connect(p=0.4)  # Dense connection
    net.add(input_group)
    net.add(input_syn)

    net.run(DURATION)
    reservoir_states.append(mon.count[:])

    if i % 20 == 0:
        print(f" > Listened to {i} vowels...")

# 4. Train & Test
split = int(len(X_mfcc) * 0.8)  # 80 Train / 20 Test
decoder.train(reservoir_states[:split], y[:split])

print("\n--- FINAL SPEECH EXAM ---")
correct = 0
tests = reservoir_states[split:]
labels = y[split:]

for i, state in enumerate(tests):
    pred = decoder.predict(state)[0]
    p_label = 0 if pred < 0.5 else 1
    target = labels[i]
    word = "AHH" if target == 0 else "EEE"

    status = "✅ PASS" if p_label == target else "❌ FAIL"
    print(f"Heard: {word} | Pred: {pred:.3f} | {status}")
    if p_label == target:
        correct += 1

print(f"\nSpeech Accuracy: {(correct/len(tests))*100}%")
