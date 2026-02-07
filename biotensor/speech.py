import numpy as np
import scipy.signal


class VowelSynthesizer:
    """
    Generates synthetic human vowel sounds using Formant Synthesis.
    """

    def __init__(self, sr=16000):
        self.sr = sr

    def generate_vowel(self, vowel_type, duration=0.5):
        """
        Creates a vowel sound ('a' or 'i').
        """
        t = np.linspace(0, duration, int(self.sr * duration))

        # 1. The "Glottal Source" (The buzzing of vocal cords)
        # A fundamental frequency (Pitch) of 120 Hz (Male voice)
        f0 = 120
        source = scipy.signal.sawtooth(2 * np.pi * f0 * t)

        # 2. Apply Formants (The shape of the mouth)
        # These are the resonant frequencies that define the vowel.
        if vowel_type == 'a':  # "Ahh" (Father)
            formants = [730, 1090, 2440]
        elif vowel_type == 'i':  # "Eee" (See)
            formants = [270, 2290, 3010]
        else:
            raise ValueError("Unknown vowel")

        # 3. Filter the source through the "Mouth"
        output = source
        for f in formants:
            # Create a resonator (Bandpass filter) for each formant
            b, a = scipy.signal.iirpeak(f, Q=10, fs=self.sr)
            output = scipy.signal.lfilter(b, a, output)

        # Normalize volume
        output = output / np.max(np.abs(output))
        return output


# --- TEST ---
if __name__ == "__main__":
    import soundfile as sf
    synth = VowelSynthesizer()

    # Generate and save to listen
    print("Generating 'Ahh'...")
    wav_a = synth.generate_vowel('a')
    sf.write('test_ahh.wav', wav_a, 16000)

    print("Generating 'Eee'...")
    wav_i = synth.generate_vowel('i')
    sf.write('test_eee.wav', wav_i, 16000)
    print("Done! Open the WAV files to hear the robot voice.")
