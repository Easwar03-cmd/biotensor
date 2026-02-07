import numpy as np
from sklearn.linear_model import Ridge


class ReadoutLayer:
    """
    The Translator: Decodes the chaotic ripples into a clear answer.
    """

    def __init__(self, regularization=1.0):
        # We use Ridge Regression (Linear Readout)
        self.model = Ridge(alpha=regularization)

    def train(self, reservoir_states, target_labels):
        """
        Learns to associate Ripple Patterns (X) with Labels (Y).
        """
        self.model.fit(reservoir_states, target_labels)
        print("BioTensor Decoder Trained.")

    def predict(self, reservoir_state):
        """
        Guesses the label for a new pattern.
        """
        # Reshape for single sample prediction
        state = np.array(reservoir_state).reshape(1, -1)
        return self.model.predict(state)
