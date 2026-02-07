import librosa
import numpy as np
import matplotlib.pyplot as plt


class AudioProcessor:
    """
    The 'Ears' of the BioTensor system.
    Converts raw audio waves into spike-ready spectrograms.
    """

    def __init__(self, sample_rate=16000, n_mfcc=20):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc  # Number of frequency bands (Input Channels)

    def process_file_from_array(self, audio_array):
        """
        Processes raw audio data (numpy array) directly.
        """
        # Pad if short
        target_len = int(self.sample_rate * 1.0)  # 1 second
        if len(audio_array) < target_len:
            audio_array = np.pad(
                audio_array, (0, target_len - len(audio_array)))
        else:
            audio_array = audio_array[:target_len]

        # Extract MFCC
        mfccs = librosa.feature.mfcc(
            y=audio_array, sr=self.sample_rate, n_mfcc=self.n_mfcc)

        # Normalize (Drop the 0th coefficient to fix the "Purple/Yellow" issue)
        mfccs = mfccs[1:]  # <--- THIS FIXES YOUR VISUALIZATION ISSUE

        mfcc_min = np.min(mfccs)
        mfcc_max = np.max(mfccs)
        mfccs_norm = (mfccs - mfcc_min) / (mfcc_max - mfcc_min + 1e-8)

        return mfccs_norm

    def visualize(self, mfcc_data):
        """
        Debug tool to see what the ear hears.
        """
        plt.figure(figsize=(10, 4))
        plt.imshow(mfcc_data, aspect='auto', origin='lower', cmap='viridis')
        plt.colorbar(label='Intensity')
        plt.title(f"Auditory Input Pattern ({self.n_mfcc} Channels)")
        plt.xlabel("Time Steps")
        plt.ylabel("Frequency Band (Low -> High)")
        plt.show()


# --- SELF-TEST ---
if __name__ == "__main__":
    # Create a dummy test file if none exists
    import soundfile as sf
    dummy_audio = np.random.uniform(-1, 1, 16000)
    sf.write('test_audio.wav', dummy_audio, 16000)

    print("Testing Audio Processor...")
    processor = AudioProcessor()
    data = processor.process_file('test_audio.wav')
    print(f"Output Shape: {data.shape} (Bands x Time)")
    processor.visualize(data)
