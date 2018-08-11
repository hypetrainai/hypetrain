import librosa
import librosa.display
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def Convert16BitToFloat(*data):
    return [d / 2**15 for d in data]


# Not fully exact but good enough for debugging uses.
def FramesToSamples(num_frames, sr=44100, hop_length_ms=10):
    hop_length = int(hop_length_ms / 1000 * sr)
    return num_frames * hop_length


def PlotMel(title, mel, sr=44100, hop_length_ms=10, fmin=25):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(
        mel,
        sr=sr,
        hop_length=int(hop_length_ms / 1000 * sr),
        fmin=fmin,
        x_axis='time',
        cmap='viridis')
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()


def MelSpectrogram(waveform,
                   sr=44100,
                   hop_length_ms=10,
                   n_mels=128,
                   fmin=25,
                   mel_magnitude_lower_bound=1e-3):
    hop_length = int(hop_length_ms / 1000 * sr)
    n_fft = hop_length * 4

    mel = librosa.feature.melspectrogram(
        waveform,
        sr=sr,
        hop_length=hop_length,
        n_fft=n_fft,
        power=1,
        n_mels=n_mels,
        fmin=fmin)
    return np.log(np.maximum(mel, mel_magnitude_lower_bound))
