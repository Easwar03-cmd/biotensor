import numpy as np
from biotensor.vision import VisionLoader
from biotensor.decoder import ReadoutLayer
from biotensor.reservoir import BioReservoir
from biotensor.encoder import SpikeEncoder
from brian2 import *
import sys
import os
sys.path.append(os.getcwd())


# --- SETUP ---
seed_val = 42
N_NEURONS = 1000
DURATION = 200*ms
loader = VisionLoader()
X, y = loader.get_pair(digit_A=0, digit_B=1, n_samples=30)  # 30 samples


def test_configuration(name, threshold_val, refractory_val):
    print(f"\n--- TESTING CONFIG: {name} ---")
    print(
        f"   [Parameters] Threshold: {threshold_val} | Refractory: {refractory_val}")

    # 1. Initialize Stack with SPECIFIC parameters
    encoder = SpikeEncoder(duration=DURATION, max_freq=100*Hz)
    decoder = ReadoutLayer(regularization=2.0)
    reservoir_states = []

    total_energy_cost = 0  # Track total spikes

    # 2. Run Simulation Loop
    for i, image in enumerate(X):
        # Reset Seed for fairness
        seed(seed_val)
        np.random.seed(seed_val)

        start_scope()

        # --- CUSTOM BIO-RESERVOIR BUILD ---
        # We manually build it here to inject the custom parameters
        eqs = '''
        dv/dt = (I - v) / (30*ms) : 1 (unless refractory)
        dI/dt = -I / (5*ms) : 1 
        '''
        neurons = NeuronGroup(N_NEURONS, eqs,
                              threshold=f'v>{threshold_val}', reset='v=0',
                              refractory=refractory_val, method='exact')
        neurons.v = 0
        synapses = Synapses(neurons, neurons, on_pre='I += 0.5')
        synapses.connect(p=0.1)
        spike_monitor = SpikeMonitor(neurons)
        net = Network(neurons, synapses, spike_monitor)

        # Connect Input (Standardized)
        input_group = encoder.encode_image(image)
        input_syn = Synapses(input_group, neurons, on_pre='v += 0.4')
        input_syn.connect(p=0.3)
        net.add(input_group)
        net.add(input_syn)

        # Run
        net.run(DURATION)

        # Collect Data
        state = spike_monitor.count[:]
        reservoir_states.append(state)
        total_energy_cost += np.sum(state)

    # 3. Train & Test
    split = int(len(X) * 0.8)
    decoder.train(reservoir_states[:split], y[:split])

    correct = 0
    test_states = reservoir_states[split:]
    test_labels = y[split:]

    for i, state in enumerate(test_states):
        pred = decoder.predict(state)[0]
        label = 0 if pred < 0.5 else 1
        if label == test_labels[i]:
            correct += 1

    accuracy = (correct / len(test_states)) * 100
    avg_energy = int(total_energy_cost / len(X))

    print(
        f"   [Result] Accuracy: {accuracy:.1f}% | Avg Spikes/Image: {avg_energy}")
    return accuracy, avg_energy


# --- MAIN EXPERIMENT ---
print("STARTING OPTIMIZATION SWEEP...")

# Config 1: What we have now (High Energy)
acc_1, cost_1 = test_configuration(
    "The Sprinter", threshold_val=1.2, refractory_val=5*ms)

# Config 2: Harder to fire (Medium Energy)
acc_2, cost_2 = test_configuration(
    "The Jogger", threshold_val=1.5, refractory_val=10*ms)

# Config 3: Very strict (Low Energy)
acc_3, cost_3 = test_configuration(
    "The Monk", threshold_val=2.0, refractory_val=20*ms)

print("\n--- FINAL REPORT CARD ---")
print(f"1. Sprinter: {acc_1}% acc | {cost_1} spikes (Baseline)")
print(f"2. Jogger:   {acc_2}% acc | {cost_2} spikes")
print(f"3. Monk:     {acc_3}% acc | {cost_3} spikes")

# Recommendation Engine
best_config = ""
if acc_3 == 100:
    best_config = "The Monk (Maximum Efficiency)"
elif acc_2 == 100:
    best_config = "The Jogger (Balanced)"
else:
    best_config = "The Sprinter (Accuracy First)"

print(f"\n🏆 RECOMMENDED ARCHITECTURE: {best_config}")
