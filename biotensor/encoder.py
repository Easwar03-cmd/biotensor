from brian2 import *
import numpy as np


class SpikeEncoder:
    """
    The Bridge: Converts digital data (Arrays) into Biological Spikes.
    """

    def __init__(self, duration=100*ms, max_freq=100*Hz):
        self.duration = duration
        self.max_freq = max_freq

    def encode_image(self, pixel_data):
        """
        Takes a 1D array of pixel brightness (0.0 to 1.0)
        Returns a PoissonGroup ready for the reservoir.
        """
        # Ensure data is normalized (0 to 1)
        pixel_data = np.clip(pixel_data, 0, 1)

        # Create the spike source
        # Biology Rule: Brighter pixel = Faster firing rate
        input_group = PoissonGroup(
            len(pixel_data), rates=pixel_data * self.max_freq)

        return input_group

    def encode_value(self, value):
        """
        Encodes a single float value into a single neuron spike train.
        """
        return self.encode_image(np.array([value]))
