from brian2 import *


class BioReservoir:
    def __init__(self, n_neurons=1000, connection_prob=0.1):
        self.n_neurons = n_neurons

        # 1. Biological Equation
        # We increase the leak rate (10ms -> 5ms). The neurons forget faster.
        # This prevents "runaway excitation."
        self.eqs = '''
        dv/dt = (I - v) / (30*ms) : 1 (unless refractory)
        dI/dt = -I / (5*ms) : 1 
        '''

        # 2. Neurons
        # Refractory period 5ms. A neuron can only fire max 200 times a second now.
        self.neurons = NeuronGroup(self.n_neurons, self.eqs,
                                   threshold='v>1.2', reset='v=0',
                                   refractory=5*ms, method='exact')
        self.neurons.v = 0

        # 3. Internal Wiring
        # Weak internal connections (0.5)
        self.synapses = Synapses(self.neurons, self.neurons, on_pre='I += 0.5')
        self.synapses.connect(p=connection_prob)

        self.spike_monitor = SpikeMonitor(self.neurons)
        self.net = Network(self.neurons, self.synapses, self.spike_monitor)

    def connect_input(self, input_group):
        # 4. Input Wiring (The Whisper)
        # We lower this to 0.4. It requires 3-4 active pixels to trigger a neuron.
        self.input_synapse = Synapses(
            input_group, self.neurons, on_pre='v += 0.4')
        self.input_synapse.connect(p=0.3)

        self.net.add(input_group)
        self.net.add(self.input_synapse)

    def run_simulation(self, duration=100*ms):
        self.net.run(duration)

    def get_state(self):
        return self.spike_monitor.count[:]
