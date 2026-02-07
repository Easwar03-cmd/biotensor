import numpy as np
from biotensor.decoder import ReadoutLayer
from biotensor.reservoir import BioReservoir
from biotensor.encoder import SpikeEncoder
from brian2 import *
import sys
import os
sys.path.append(os.getcwd())

# 1. SETUP
start_scope()  # Reset Brian2 clock
print("Initializing BioTensor Stack...")

# Define Data (Example: 2 pixels. [0,1] is Pattern A, [1,0] is Pattern B)
data_A = np.array([0.1, 0.9])
data_B = np.array([0.9, 0.1])
target_A = 0
target_B = 1

# 2. INITIALIZE MODULES
encoder = SpikeEncoder(duration=200*ms)
organoid = BioReservoir(n_neurons=500)
decoder = ReadoutLayer()

# 3. RUN SIMULATION FOR DATA A
print("Processing Pattern A...")
input_A = encoder.encode_image(data_A)
organoid.connect_input(input_A)
organoid.run_simulation()
state_A = organoid.get_state()  # Capture the ripple

# 4. RESET & RUN SIMULATION FOR DATA B
print("Processing Pattern B...")
start_scope()  # Important: Reset time for the next run
organoid = BioReservoir(n_neurons=500)  # Re-build fresh brain
input_B = encoder.encode_image(data_B)
organoid.connect_input(input_B)
organoid.run_simulation()
state_B = organoid.get_state()

# 5. TRAIN DECODER
# We stack the states to create a dataset
X = np.vstack([state_A, state_B])
Y = np.array([target_A, target_B])

print("Training Readout Layer...")
decoder.train(X, Y)

# 6. TEST
print("Testing...")
prediction = decoder.predict(state_A)
print(
    f"Input: Pattern A | Prediction: {prediction[0]:.4f} | Target: {target_A}")

if abs(prediction[0] - target_A) < 0.1:
    print("SUCCESS: BioTensor correctly identified the pattern.")
else:
    print("FAIL: Retrain needed.")
