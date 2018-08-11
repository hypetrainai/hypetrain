import librosa
import librosa.display
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def Convert16BitToFloat(*data):
    return [d / 2**15 for d in data]


def ConvertFloatTo16Bit(*data):
    return [d * 2**15 for d in data]


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


def STFT(waveform, sr=44100, hop_length_ms=10):
    hop_length = int(hop_length_ms / 1000 * sr)
    n_fft = hop_length * 4
    return librosa.core.stft(
        waveform, hop_length=hop_length, n_fft=n_fft, center=False)


def InverseSTFT(stft, sr=44100, hop_length_ms=10):
    hop_length = int(hop_length_ms / 1000 * sr)
    n_fft = hop_length * 4
    return librosa.core.istft(
        stft, hop_length=hop_length, win_length=n_fft, center=False)


def MelSpectrogram(waveform,
                   sr=44100,
                   hop_length_ms=10,
                   n_mels=128,
                   fmin=25,
                   mel_magnitude_lower_bound=1e-3):
    hop_length = int(hop_length_ms / 1000 * sr)
    n_fft = hop_length * 4

    stft = STFT(waveform, sr, hop_length_ms)
    mel = librosa.feature.melspectrogram(
        S=np.abs(stft),
        sr=sr,
        hop_length=hop_length,
        n_fft=n_fft,
        power=1,
        n_mels=n_mels,
        fmin=fmin)
    num_frames = (len(waveform) - n_fft) // hop_length + 1
    assert mel.shape[1] == num_frames, mel.shape

    return np.log(np.maximum(mel, mel_magnitude_lower_bound))
