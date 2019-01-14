import torch
import os
import librosa
import librosa.display
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from GLOBALS import FLAGS


def SaveModel(step, model, optimizer, ckpt_prefix='model'):
    print('Saving model!')
    model_state = {
        'step': step,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(model_state, FLAGS.log_dir + '/%s_%d.pt' % (ckpt_prefix, step))
    torch.save(model_state, FLAGS.log_dir + '/%s_latest.pt' % ckpt_prefix)
    print('Model saved!')


def LoadModel(model, optimizer=None, ckpt_prefix='model'):
    try:
        filename = os.path.join(FLAGS.log_dir, '%s_%s.pt' % (ckpt_prefix, FLAGS.checkpoint))
        print('Restoring from %s' % filename)
        model_state = torch.load(filename)
    except FileNotFoundError:
        input('Could not find checkpoint to restore! Press enter to continue from scratch.')
        return 0
    model.load_state_dict(model_state['state_dict'])
    if optimizer:
        optimizer.load_state_dict(model_state['optimizer'])
    return model_state['step']


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


def PlotSpectrogram(title, spectrogram, y_axis='linear'):
    fig = plt.figure(figsize=(10, 4))
    librosa.display.specshow(
        spectrogram,
        sr=FLAGS.sample_rate,
        hop_length=MillisecondsToSamples(FLAGS.hop_length_ms),
        fmin=FLAGS.fmin,
        x_axis='time',
        y_axis=y_axis,
        cmap='viridis')
    plt.colorbar()
    plt.clim(0, 4);
    plt.title(title)
    plt.tight_layout()
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    data = data.transpose([2, 0, 1])
    plt.close()
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

    n_fft, _, _ = NFFT()
    mel_matrix = librosa.filters.mel(FLAGS.sample_rate, n_fft, FLAGS.n_mels, FLAGS.fmin)
    mel = np.dot(mel_matrix, np.abs(stft)**2)
    return (1 + mel)**(1./3.) - 1


def InverseMelSpectrogram(mel_spectrogram):
    """Returns stft magnitudes. Follow up with InverseSTFT for waveforms."""
    assert np.all(mel_spectrogram >= 0.0)
    mel_spectrogram = (mel_spectrogram + 1)**3 - 1

    n_fft, _, _ = NFFT()
    mel_matrix = librosa.filters.mel(FLAGS.sample_rate, n_fft, FLAGS.n_mels, FLAGS.fmin)
    inv_mel_matrix = np.linalg.pinv(mel_matrix)
    linear_spectrogram = np.dot(inv_mel_matrix, mel_spectrogram)
    return np.sqrt(np.maximum(0.0, linear_spectrogram))
