import numpy as np
from biotensor.vision import VisionLoader
from biotensor.decoder import ReadoutLayer
from biotensor.reservoir import BioReservoir
from biotensor.encoder import SpikeEncoder
from brian2 import *
import sys
import os
sys.path.append(os.getcwd())


# --- CONFIGURATION ---
seed_value = 42
N_NEURONS = 500  # Smaller brain is easier to debug
DURATION = 200*ms

print("--- BIOTENSOR: BALANCED MODE ---")

# 1. LOAD DATA
loader = VisionLoader()
X, y = loader.get_pair(digit_A=0, digit_B=1, n_samples=40)

# 2. INITIALIZE
encoder = SpikeEncoder(duration=DURATION, max_freq=100*Hz)
# Higher regularization helps generalization
decoder = ReadoutLayer(regularization=5.0)

reservoir_states = []

print(f"Processing {len(X)} images...")

for i, image in enumerate(X):
    # Reset Seed
    seed(seed_value)
    np.random.seed(seed_value)

    start_scope()
    organoid = BioReservoir(n_neurons=N_NEURONS)

    # Run
    input_spikes = encoder.encode_image(image)
    organoid.connect_input(input_spikes)
    organoid.run_simulation(DURATION)

    state = organoid.get_state()
    reservoir_states.append(state)

    # DEBUG: Print activity level
    spikes = np.sum(state)
    if i < 3:  # Print first 3 only
        print(f"Img {i} (Label {y[i]}) -> Total Spikes: {spikes}")


# --- CRITICAL CHECK: ARE THE STATES DIFFERENT? ---
diff = np.linalg.norm(reservoir_states[0] - reservoir_states[1])
print(f"\nDifference between Img 0 and Img 1: {diff:.2f}")
if diff == 0:
    print("FATAL ERROR: The brain responses are identical!")
    sys.exit()

# 3. TRAIN & TEST
split_idx = int(len(X) * 0.75)  # 30 Train, 10 Test
X_train = reservoir_states[:split_idx]
y_train = y[:split_idx]
X_test = reservoir_states[split_idx:]
y_test = y[split_idx:]

decoder.train(X_train, y_train)

print("\n--- FINAL EXAM ---")
correct = 0
for i, state in enumerate(X_test):
    prediction = decoder.predict(state)
    target = y_test[i]
    pred_label = 0 if prediction[0] < 0.5 else 1

    status = "✅ PASS" if pred_label == target else "❌ FAIL"
    print(f"Img {i} | Target: {target} | Pred: {prediction[0]:.3f} | {status}")

    if pred_label == target:
        correct += 1

print(f"\nFinal Accuracy: {(correct/len(X_test))*100}%")
