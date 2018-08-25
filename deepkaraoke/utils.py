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


def FFTChannels(window_length_ms):
    window_length = int(window_length_ms / 1000 * sr)
    n_fft = 2**np.ceil(np.log2(window_length))
    return n_fft, window_length


def STFT(waveform, sr=44100, hop_length_ms=10, window_length_ms=40):
    hop_length = int(hop_length_ms / 1000 * sr)
    n_fft, window_length = FFTChannels(window_length_ms)
    return librosa.core.stft(
        waveform,
        hop_length=hop_length,
        win_length=window_length,
        n_fft=n_fft,
        center=False)


def InverseSTFT(stft, sr=44100, hop_length_ms=10, window_length_ms=40):
    hop_length = int(hop_length_ms / 1000 * sr)
    _, window_length = FFTChannels(window_length_ms)
    return librosa.core.istft(
        stft, hop_length=hop_length, win_length=window_length, center=False)


def MelSpectrogram(waveform_or_stft,
                   sr=44100,
                   hop_length_ms=10,
                   window_length_ms=40,
                   n_mels=128,
                   fmin=25,
                   mel_magnitude_lower_bound=1e-3):
    if len(waveform_or_stft.shape) == 1:
        stft = STFT(waveform_or_stft, sr, hop_length_ms, window_length_ms)
    else:
        stft = waveform_or_stft

    n_fft, _ = FFTChannels(window_length_ms)
    mel_matrix = librosa.filters.mel(sr, n_fft, n_mels, fmin).T
    mel = np.dot(np.abs(stft), mel_matrix)
    return np.log(np.maximum(mel, mel_magnitude_lower_bound))


def InverseMelSpectrogram(log_mel_spectrogram,
                          sr=44100,
                          window_length_ms=40,
                          n_mels=128,
                          fmin=25):
    """Returns stft magnitudes. Follow up with InverseSTFT for waveforms."""
    mel_spectrogram = np.exp(log_mel_spectrogram)

    n_fft, _ = FFTChannels(window_length_ms)
    mel_matrix = librosa.filters.mel(sr, n_fft, n_mels, fmin).T
    row_norm = np.sum(mel_matrix, axis=0)[np.newaxis, :]
    return np.dot(mel_spectrogram / row_norm, mel_matrix.T)
