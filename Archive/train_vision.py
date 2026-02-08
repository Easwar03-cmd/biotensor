import numpy as np
from biotensor.vision import VisionLoader
from biotensor.decoder import ReadoutLayer
from biotensor.encoder import SpikeEncoder
from brian2 import *
import sys
import os
sys.path.append(os.getcwd())


# --- SETUP ---
seed_val = 42
N_NEURONS = 1000
DURATION = 250*ms  # Longer thinking time (was 200ms)
loader = VisionLoader()
X, y = loader.get_pair(digit_A=0, digit_B=1, n_samples=30)

print("--- TESTING: THE ENLIGHTENED MONK ---")
print("Strategy: Lower Threshold + Longer Memory")

# 1. Initialize Stack
encoder = SpikeEncoder(duration=DURATION, max_freq=100*Hz)
decoder = ReadoutLayer(regularization=1.5)  # Slightly looser regularization
reservoir_states = []
total_energy_cost = 0

# 2. Run Simulation
for i, image in enumerate(X):
    seed(seed_val)
    np.random.seed(seed_val)
    start_scope()

    # --- TUNED PARAMETERS ---
    # Memory (tau) increased to 50ms (was 30ms) -> "Slower forgetting"
    # Threshold lowered to 1.7 (was 2.0) -> "More sensitive"
    # Refractory set to 15ms -> "Stays calm"
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
    spike_monitor = SpikeMonitor(neurons)
    net = Network(neurons, synapses, spike_monitor)

    # Input
    input_group = encoder.encode_image(image)
    input_syn = Synapses(input_group, neurons, on_pre='v += 0.4')
    input_syn.connect(p=0.3)
    net.add(input_group)
    net.add(input_syn)

    net.run(DURATION)

    state = spike_monitor.count[:]
    reservoir_states.append(state)
    total_energy_cost += np.sum(state)

# 3. Report Results
split = int(len(X) * 0.8)
decoder.train(reservoir_states[:split], y[:split])

correct = 0
test_states = reservoir_states[split:]
test_labels = y[split:]

print("\n--- TEST RESULTS ---")
for i, state in enumerate(test_states):
    pred = decoder.predict(state)[0]
    label = 0 if pred < 0.5 else 1
    status = "✅ PASS" if label == test_labels[i] else "❌ FAIL"
    print(f"Img {i} | Target: {test_labels[i]} | Pred: {pred:.3f} | {status}")
    if label == test_labels[i]:
        correct += 1

accuracy = (correct / len(test_states)) * 100
avg_spikes = int(total_energy_cost / len(X))

print(f"\nFinal Accuracy: {accuracy}%")
print(f"Average Spikes: {avg_spikes}")

if accuracy == 100 and avg_spikes < 15000:
    print("\n🏆 SUCCESS: OPTIMIZED ARCHITECTURE FOUND.")
else:
    print("\n⚠️ ADJUSTMENT NEEDED.")
