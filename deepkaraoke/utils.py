import librosa
import librosa.display
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from CONSOLE_ARGS import ARGS as FLAGS


def Convert16BitToFloat(*data):
    ret = [d / 2**15 for d in data]
    return ret[0] if len(ret) == 1 else ret


def ConvertFloatTo16Bit(*data):
    ret = [d * 2**15 for d in data]
    return ret[0] if len(ret) == 1 else ret


def MillisecondsToSamples(length_ms):
    return int(length_ms / 1000 * FLAGS.sample_rate)


# Not fully exact but good enough for debugging uses.
def FramesToSamples(num_frames):
    return num_frames * MillisecondsToSamples(FLAGS.hop_length_ms)


def PlotMel(title, mel):
    fig = plt.figure(figsize=(10, 4))
    librosa.display.specshow(
        mel,
        sr=FLAGS.sample_rate,
        hop_length=MillisecondsToSamples(FLAGS.hop_length_ms),
        fmin=FLAGS.fmin,
        x_axis='time',
        y_axis='linear',  # TODO: use mel.
        cmap='viridis')
    plt.colorbar()
    plt.clim(0, 4);
    plt.title(title)
    plt.tight_layout()
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    data = data.transpose([2, 0, 1])
    return data


def NFFT():
    window_length = MillisecondsToSamples(FLAGS.window_length_ms)
    n_fft = int(2**np.ceil(np.log2(window_length)))
    fft_channels = (n_fft + 1) // 2 + 1
    return n_fft, fft_channels, window_length


def STFT(waveform):
    n_fft, _, window_length = NFFT()
    return librosa.core.stft(
        waveform,
        hop_length=MillisecondsToSamples(FLAGS.hop_length_ms),
        win_length=window_length,
        n_fft=n_fft,
        center=False)


def InverseSTFT(stft):
    _, _, window_length = NFFT()
    return librosa.core.istft(
        stft,
        hop_length=MillisecondsToSamples(FLAGS.hop_length_ms),
        win_length=window_length,
        center=False)


def MelSpectrogram(waveform_or_stft):
    if len(waveform_or_stft.shape) == 1:
        stft = STFT(waveform_or_stft)
    else:
        stft = waveform_or_stft

    # TODO: Use mel.
    # n_fft, _, _ = NFFT()
    # mel_matrix = librosa.filters.mel(FLAGS.sample_rate, n_fft, FLAGS.n_mels, FLAGS.fmin)
    # mel = np.dot(mel_matrix, np.abs(stft)**2)
    mel = np.abs(stft)**2
    return mel**(1./3.)


def InverseMelSpectrogram(mel_spectrogram):
    """Returns stft magnitudes. Follow up with InverseSTFT for waveforms."""
    mel_spectrogram = mel_spectrogram**3

    # TODO: Use mel.
    # n_fft, _, _ = NFFT()
    # mel_matrix = librosa.filters.mel(FLAGS.sample_rate, n_fft, FLAGS.n_mels, FLAGS.fmin)
    # inv_mel_matrix = np.linalg.pinv(mel_matrix)
    # linear_spectrogram = np.dot(inv_mel_matrix, mel_spectrogram)
    linear_spectrogram = mel_spectrogram
    return np.sqrt(np.maximum(0, linear_spectrogram))
