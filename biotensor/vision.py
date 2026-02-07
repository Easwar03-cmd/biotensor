from sklearn.datasets import load_digits
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np


class VisionLoader:
    """
    Helper to load real-world image data for the Organoid.
    """

    def __init__(self):
        # Load the famous "Digits" dataset (Handwritten numbers 0-9)
        self.digits = load_digits()

        # Normalize data: standard AI images are 0-255.
        # Neurons need 0.0-1.0 to fire correctly.
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def get_pair(self, digit_A, digit_B, n_samples=20):
        """
        Extracts only two specific numbers (e.g., all the '0's and '1's).
        We start with binary classification because it's easier for the organoid.
        """
        # Filter the data
        mask = (self.digits.target == digit_A) | (
            self.digits.target == digit_B)
        X_filtered = self.digits.data[mask]
        y_filtered = self.digits.target[mask]

        # Scale the pixel brightness
        X_scaled = self.scaler.fit_transform(X_filtered)

        # Convert labels to 0 and 1 (Binary)
        # If target is digit_A -> 0. If target is digit_B -> 1.
        y_binary = np.where(y_filtered == digit_A, 0, 1)

        # Return a small batch for testing
        return X_scaled[:n_samples], y_binary[:n_samples]

    def show_sample(self, image_data):
        """
        Debug tool to see what the Organoid sees.
        """
        import matplotlib.pyplot as plt
        plt.imshow(image_data.reshape(8, 8), cmap='gray')
        plt.title("Input Image")
        plt.show()
