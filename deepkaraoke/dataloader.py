import collections
import glob
import numpy as np
import os
import struct
import utils
import wave
from GLOBALS import FLAGS


class KaraokeDataLoader(object):
    """Dataloader for karaoke task.

    Args:
        data_dir: a directory that contains .wav files. Each wav file should
                  be int16 PCM and have 2 channels [instruments, vocals].
        batch_size: output batch size.
        ignore_percentage: the area on either end of the song to ignore.
    """

    def __init__(self,
                 data_dir,
                 batch_size=24,
                 ignore_percentage=0.05):
        self.batch_size = batch_size
        self.ignore_percentage = ignore_percentage

        self.filenames = glob.glob(os.path.join(data_dir, '*.wav'))
        self.N = len(self.filenames)
        self.song_lengths = {}
        for filename in self.filenames:
            f = wave.open(filename, 'r')
            assert f.getnchannels() == 2, filename  # instruments, vocals
            assert f.getsampwidth() == 2, filename  # int16 data
            assert f.getframerate() == FLAGS.sample_rate, filename
            self.song_lengths[filename] = f.getnframes()
            f.close()

    def extract_data(self, filename, start_index, sample_length):
        f = wave.open(filename, 'r')
        f.setpos(start_index)
        data = struct.unpack('%ih' % (sample_length * 2), f.readframes(sample_length))
        f.close()
        # Stereo data is interleaved.
        data = np.reshape(np.array(data), (-1, 2))
        # Convert to float.
        data = (data / 32768.).astype(np.float32)
        return data[:, 0], data[:, 1]

    def get_random_batch(self, sample_length, batch_size=None):
        batch_size = batch_size or self.batch_size
        names = np.random.choice(self.filenames, batch_size)
        lengths = np.array([self.song_lengths[name] for name in names])
        if sample_length == -1:
            starts = [0] * batch_size
            sample_lengths = lengths
        else:
            max_start_offsets = (
                (1 - 2 * self.ignore_percentage) * lengths - sample_length)
            start_offsets = np.random.rand(batch_size) * max_start_offsets
            starts = (
                self.ignore_percentage * lengths + start_offsets).astype(int)
            sample_lengths = [sample_length] * batch_size
        return [
            self.extract_data(name, start, length)
            for name, start, length in
            zip(names, starts, sample_lengths)
        ]

    def get_single_segment(self, extract_idx=0, start_index=3000000, sample_length=200000):
        name = self.filenames[extract_idx]
        if sample_length == -1:
            sample_length = self.song_lengths[name]
        return self.extract_data(name, start_index, sample_length)

