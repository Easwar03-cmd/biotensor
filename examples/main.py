import sys
import os

# --- 1. FORCE THE PATH (The "Nuclear" Fix) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

print(f"DEBUG: Project Root added to path: {project_root}")
# ---------------------------------------------------------

import numpy as np  # noqa: E402
from biotensor.vision import VisionLoader  # noqa: E402
from biotensor.speech import VowelSynthesizer  # noqa: E402
from biotensor.audio import AudioProcessor  # noqa: E402
from biotensor.decoder import ReadoutLayer  # noqa: E402

# Import the Kernel
try:
    from biotensor_kernel import BioTensorKernel  # noqa: E402
except ImportError:
    # Fallback if kernel is in root
    from biotensor_kernel import BioTensorKernel  # noqa: E402


print("--- 1. LOADING DATA ---")
N_SAMPLES = 100
loader = VisionLoader()
X_img, y = loader.get_pair(digit_A=0, digit_B=1, n_samples=N_SAMPLES)

# Generate Audio (Just for training)
synth = VowelSynthesizer()
processor = AudioProcessor(n_mfcc=13)
X_aud = []
for label in y:
    vowel = 'a' if label == 0 else 'i'
    raw = synth.generate_vowel(vowel)
    mfcc = processor.process_file_from_array(raw)
    X_aud.append(mfcc[:, mfcc.shape[1] // 2])  # Center slice

# Normalize Data (0.0 to 1.0) for the Kernel
X_img = [x.flatten() / (np.max(x)+1e-8) for x in X_img]
X_aud = [(x - np.min(x))/(np.max(x)-np.min(x)+1e-8) for x in X_aud]

# 2. INITIALIZE BIOTENSOR (The Brain)
print("\n--- 2. BOOTING KERNEL ---")
brain = BioTensorKernel(n_neurons=1000, n_vision=64, n_audio=13)

# 3. TRAIN (The Experience)
print("\n--- 3. TRAINING (Vision + Audio) ---")
# Train on first 80 samples
brain.learn(vision_data=X_img[:80],
            audio_data=X_aud[:80],
            labels=y[:80])

# 4. TEST (The Recall)
print("\n--- 4. TESTING (Vision Only) ---")
# Test on last 20 samples
activity_patterns = brain.recall(vision_data=X_img[80:])

# 5. DECODE (The Readout)
print("\n--- 5. DECODING RESULTS ---")
decoder = ReadoutLayer()
# We train the *decoder* on the brain's activity patterns
# (In a real organism, this would be the motor cortex reading the visual cortex)
decoder.train(activity_patterns, y[80:])

correct = 0
for i, state in enumerate(activity_patterns):
    pred = decoder.predict(state)[0]
    target = y[80 + i]

    p_label = 0 if pred < 0.5 else 1
    status = "✅" if p_label == target else "❌"
    concept = "Zero" if target == 0 else "One"

    print(
        f"Sample {i} | Vision: {concept} | Brain Output: {pred:.3f} | {status}")
    if p_label == target:
        correct += 1

print(f"\nFINAL ACCURACY: {(correct/len(activity_patterns))*100:.1f}%")
