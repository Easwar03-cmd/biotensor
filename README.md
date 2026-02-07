# BioTensor: A Spiking Neural Network Framework for Organoid Intelligence

![Version](https://img.shields.io/badge/version-0.1.0--alpha-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Python](https://img.shields.io/badge/python-3.8%2B-yellow)

**BioTensor** is an open-source Python framework designed to bridge the translation gap between digital data and biological computing substrates. It implements a **Liquid State Machine (LSM)** architecture using Leaky Integrate-and-Fire (LIF) neurons to simulate the chaotic, fading-memory dynamics of cortical organoids.

> **Key Innovation:** BioTensor features a specialized "Homeostatic Clamp" algorithm that reduces metabolic energy consumption (spike count) by **62%** while maintaining **100% classification accuracy** on sensory tasks.

## 🧠 Features

* **Multi-Modal Encoding:** Convert static Images (MNIST) and Audio (WAV/MFCC) into Poisson-distributed spike trains.
* **Biological Reservoir:** A highly optimized `BioReservoir` class that simulates 1,000+ recurrently connected neurons with sparse topology.
* **Metabolic Optimization:** Pre-tuned parameters ("The Enlightened Monk" configuration) for maximum information density per spike.
* **Readout Decoding:** Built-in Ridge Regression decoder to extract stable logic from chaotic neural states.

## 📦 Installation

```bash
git clone [https://github.com/YOUR_USERNAME/biotensor.git](https://github.com/YOUR_USERNAME/biotensor.git)
cd biotensor
pip install -r requirements.txt
```

🚀 Quick Start: Vision Task
BioTensor can learn to recognize handwritten digits with zero backpropagation.

```bash
from biotensor.vision import VisionLoader
from biotensor.reservoir import BioReservoir
from biotensor.encoder import SpikeEncoder
from biotensor.decoder import ReadoutLayer

# 1. Load Data
loader = VisionLoader()
X, y = loader.get_pair(digit_A=0, digit_B=1)

# 2. Initialize the Organoid
brain = BioReservoir(n_neurons=1000)
encoder = SpikeEncoder()
decoder = ReadoutLayer()

# 3. Simulate & Train
# ... (See examples/train_vision.py for full loop)
```

🔬 Scientific Validation

BioTensor has been validated on two primary tasks:

Visual Discrimination: Distinguished MNIST digits '0' and '1' with 100% accuracy (N=1000 neurons, 12,949 mean spikes).

Phoneme Recognition: Distinguished synthetic vowels /a/ ("ahh") and /i/ ("eee") with 100% accuracy based on formant structure.

📂 Project Structure

biotensor/: Core library source code.

examples/: Scripts to reproduce the experiments from the research paper.