import numpy as np
from brian2 import *
prefs.codegen.target = 'numpy'  # Safe Mode for compatibility


class BioTensorKernel:
    def __init__(self, n_neurons=1000, n_vision=64, n_audio=13, seed_val=2026):
        """
        Initialize the Biological Operating System.
        """
        self.N = n_neurons
        self.N_VIS = n_vision
        self.N_AUD = n_audio
        self.TOTAL_CHANNELS = n_vision + n_audio

        # Anatomy: Partition the brain
        self.midpoint = n_neurons // 2

        # Physics Constants
        self.tau_max = 50*ms
        self.duration = 100*ms

        # State: The Brain's Memory (Weights)
        # We start with random potential, effectively 'Tabula Rasa'
        np.random.seed(seed_val)
        self.weights = np.random.uniform(
            0.2, 0.5, (self.TOTAL_CHANNELS, self.N))
        self.scaling_factors = np.ones(self.N)  # For Energy Normalization

        print(
            f"[BioTensor] Kernel Booted. Cortex: {self.N} Neurons. Input: {self.TOTAL_CHANNELS} Channels.")

    def _encode_latency(self, data_vec, modality='vision'):
        """
        Convert raw data into precise spike times (Brighter/Louder = Faster).
        """
        indices = []
        times = []

        # 1. Energy Normalization (The "Fair Fight" Logic)
        if modality == 'vision':
            active_pixels = np.count_nonzero(data_vec > 0.1)
            if active_pixels == 0:
                active_pixels = 1
            energy_factor = 40.0 / active_pixels  # Boost weak signals
            data_vec = data_vec * energy_factor
            offset = 0
        else:
            # Audio is already dense, usually needs less boosting
            offset = self.N_VIS

        # Clip and Normalize
        data_vec = np.clip(data_vec, 0, 1.0)

        # 2. Latency Encoding
        for i, val in enumerate(data_vec):
            if val > 0.1:  # Threshold to ignore silence/black
                # Higher Value = Earlier Time (1.0 -> 0ms, 0.1 -> 50ms)
                spike_time = (1.0 - val) * self.tau_max
                indices.append(offset + i)
                times.append(spike_time)

        return indices, times

    def learn(self, vision_data, audio_data, labels, epochs=1):
        """
        The 'Training Mode'. Uses STDP + The Eraser to self-organize.
        """
        print(
            f"[BioTensor] Starting Learning Phase ({len(vision_data)} samples)...")

        # Standard Neuron Equation (Plasticity Enabled)
        eqs = '''
        dv/dt = (I - v + I_teach) / (20*ms) : 1 (unless refractory)
        dI/dt = -I / (5*ms) : 1 
        I_teach : 1 
        '''

        for epoch in range(epochs):
            for idx, (img, aud, label) in enumerate(zip(vision_data, audio_data, labels)):

                # A. Encode Inputs
                i_vis, t_vis = self._encode_latency(img, 'vision')
                i_aud, t_aud = self._encode_latency(aud, 'audio')

                all_indices = np.concatenate([i_vis, i_aud])
                all_times = np.concatenate([t_vis, t_aud])

                # Sort for Brian2
                if len(all_indices) > 0:
                    ord = np.argsort(all_times)
                    all_indices = all_indices[ord]
                    all_times = all_times[ord] * second
                else:
                    all_indices = [0]
                    all_times = [0]*ms

                # B. Setup Simulation
                start_scope()
                neurons = NeuronGroup(self.N, eqs, threshold='v>1.5', reset='v=0',
                                      refractory=3*ms, method='exact')
                neurons.v = 0

                # C. Teacher Signal (The Guide)
                # If Label=0 (Zero/Ahh) -> Excite Left Brain, Inhibit Right Brain
                if label == 0:
                    neurons.I_teach[:self.midpoint] = 1.5
                    neurons.I_teach[self.midpoint:] = -1.0
                else:
                    neurons.I_teach[:self.midpoint] = -1.0
                    neurons.I_teach[self.midpoint:] = 1.5

                # D. Input Injection
                input_group = SpikeGeneratorGroup(
                    self.TOTAL_CHANNELS, all_indices, all_times)
                mon_in = SpikeMonitor(input_group)
                mon_out = SpikeMonitor(neurons)

                syn = Synapses(input_group, neurons, 'w : 1', on_pre='v += w')
                syn.connect()
                syn.w = self.weights.flatten()  # Load current memory

                run(self.duration)

                # E. The Learning Rule (STDP + Eraser)
                spikes_in = mon_in.i[:]
                spikes_out = mon_out.i[:]
                unique_in = np.unique(spikes_in)
                unique_out = np.unique(spikes_out)

                # 1. Reward (LTP)
                self.weights[np.ix_(unique_in, unique_out)] += 0.2

                # 2. Punishment (The Eraser)
                if label == 0:
                    wrong_neurons = np.arange(self.midpoint, self.N)
                else:
                    wrong_neurons = np.arange(0, self.midpoint)
                self.weights[np.ix_(unique_in, wrong_neurons)] -= 0.1

                # 3. Clamp Weights (0 to 10)
                self.weights = np.clip(self.weights, 0.0, 10.0)

                # 4. Homeostasis (Budgeting)
                col_sums = np.sum(self.weights, axis=0)
                col_sums[col_sums == 0] = 1.0
                norm_factors = 10.0 / col_sums
                self.weights = self.weights * norm_factors[np.newaxis, :]

                if idx % 20 == 0:
                    print(f"  > Processed sample {idx}...")

        # F. Final Step: Post-Training Energy Scaling
        print("[BioTensor] Finalizing Memories (Energy Scaling)...")
        total_input = np.sum(self.weights, axis=0)
        total_input[total_input == 0] = 1.0
        target_capacity = 50.0
        self.scaling_factors = target_capacity / total_input
        self.weights = self.weights * self.scaling_factors[np.newaxis, :]
        print("[BioTensor] Learning Complete.")

    def recall(self, vision_data):
        """
        The 'Test Mode'. Vision ONLY. Returns raw brain activity.
        """
        results = []

        # Minimal equation (No Teacher)
        eqs = '''
        dv/dt = (I - v) / (20*ms) : 1 (unless refractory)
        dI/dt = -I / (5*ms) : 1 
        '''

        print(
            f"[BioTensor] Starting Recall Phase ({len(vision_data)} samples)...")

        for idx, img in enumerate(vision_data):
            # A. Encode Vision Only
            i_vis, t_vis = self._encode_latency(img, 'vision')

            if len(i_vis) > 0:
                ord = np.argsort(t_vis)
                i_vis = np.array(i_vis)[ord]
                t_vis = np.array(t_vis)[ord] * second
            else:
                i_vis = [0]
                t_vis = [0]*ms

            # B. Setup Brain
            start_scope()
            neurons = NeuronGroup(self.N, eqs, threshold='v>1.5', reset='v=0',
                                  refractory=3*ms, method='exact')
            neurons.v = 0

            # C. Connect Learned Weights
            input_group = SpikeGeneratorGroup(
                self.TOTAL_CHANNELS, i_vis, t_vis)
            syn = Synapses(input_group, neurons, 'w : 1', on_pre='v += w')
            syn.connect()
            syn.w = self.weights.flatten()

            # D. Lateral Inhibition (Winner-Take-All)
            # Left Brain hates Right Brain, and vice versa
            syn_inhib = Synapses(
                neurons, neurons, 'w_inhib : 1', on_pre='v -= w_inhib')
            syn_inhib.connect(
                condition=f'(i < {self.midpoint} and j >= {self.midpoint}) or (i >= {self.midpoint} and j < {self.midpoint})', p=0.05)
            syn_inhib.w_inhib = 2.0

            mon = SpikeMonitor(neurons)
            run(self.duration)

            results.append(mon.count[:])

        return np.array(results)
