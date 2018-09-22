import librosa
import librosa.display
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def Convert16BitToFloat(*data):
    ret = [d / 2**15 for d in data]
    return ret[0] if len(ret) == 1 else ret


def ConvertFloatTo16Bit(*data):
    ret = [d * 2**15 for d in data]
    return ret[0] if len(ret) == 1 else ret


# Not fully exact but good enough for debugging uses.
def FramesToSamples(num_frames, sr=44100, hop_length_ms=10):
    hop_length = int(hop_length_ms / 1000 * sr)
    return num_frames * hop_length


def PlotMel(title, mel, sr=44100, hop_length_ms=10, fmin=25):
    fig = plt.figure(figsize=(10, 4))
    librosa.display.specshow(
        mel,
        sr=sr,
        hop_length=int(hop_length_ms / 1000 * sr),
        fmin=fmin,
        x_axis='time',
        cmap='viridis')
    plt.colorbar()
    plt.clim(1, 6);
    plt.title(title)
    plt.tight_layout()
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    data = data.transpose([2, 0, 1])
    return data


def NFFT(sr, window_length_ms):
    window_length = int(window_length_ms / 1000 * sr)
    n_fft = int(2**np.ceil(np.log2(window_length)))
    return n_fft, window_length


def STFT(waveform, sr=44100, hop_length_ms=10, window_length_ms=40):
    hop_length = int(hop_length_ms / 1000 * sr)
    n_fft, window_length = NFFT(sr, window_length_ms)
    return librosa.core.stft(
        waveform,
        hop_length=hop_length,
        win_length=window_length,
        n_fft=n_fft,
        center=False)


def InverseSTFT(stft, sr=44100, hop_length_ms=10, window_length_ms=40):
    hop_length = int(hop_length_ms / 1000 * sr)
    _, window_length = NFFT(sr, window_length_ms)
    return librosa.core.istft(
        stft, hop_length=hop_length, win_length=window_length, center=False)


def MelSpectrogram(waveform_or_stft,
                   sr=44100,
                   hop_length_ms=10,
                   window_length_ms=40,
                   n_mels=128,
                   fmin=25):
    if len(waveform_or_stft.shape) == 1:
        stft = STFT(waveform_or_stft, sr, hop_length_ms, window_length_ms)
    else:
        stft = waveform_or_stft

    n_fft, _ = NFFT(sr, window_length_ms)
    mel_matrix = librosa.filters.mel(sr, n_fft, n_mels, fmin)
    # TODO: Use mel.
    # mel = np.dot(mel_matrix, np.abs(stft)**2)
    mel = np.abs(stft)**2
    return (1 + mel)**(1./3.)


def InverseMelSpectrogram(mel_spectrogram,
                          sr=44100,
                          window_length_ms=40,
                          n_mels=128,
                          fmin=25):
    """Returns stft magnitudes. Follow up with InverseSTFT for waveforms."""
    mel_spectrogram = mel_spectrogram**3 - 1

    n_fft, _ = NFFT(sr, window_length_ms)
    mel_matrix = librosa.filters.mel(sr, n_fft, n_mels, fmin)
    inv_mel_matrix = np.linalg.pinv(mel_matrix)
    # TODO: Use mel.
    # linear_spectrogram = np.dot(inv_mel_matrix, mel_spectrogram)
    linear_spectrogram = mel_spectrogram
    return np.sqrt(np.maximum(0, linear_spectrogram))
