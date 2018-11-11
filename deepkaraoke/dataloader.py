import collections
import numpy as np
import pickle as pkl

DataItem = collections.namedtuple('DataItem',
                                  ['name', 'start_offset', 'length', 'data'])


class KaraokeDataLoader(object):
    def __init__(self,
                 data_file,
                 batch_size=24,
                 ignore_percentage=0.05,
                 sample_length=7056):
        self.batch_size = batch_size
        #ignore_percentage defines the area on either end of the song which we want to ignore.
        self.ignore_percentage = ignore_percentage
        self.sample_length = sample_length

        with open(data_file, 'rb') as file:
            self.data = pkl.load(file)
            self.N = len(self.data)
            self.song_lengths = {k: len(d[0]) for k, d in self.data.items()}

    def get_random_batch(self, sample_length=None, batch_size=None):
        sample_length = sample_length or self.sample_length
        batch_size = batch_size or self.batch_size
        names = np.random.choice(list(self.data.keys()), batch_size)
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
            DataItem(
                name=name,
                start_offset=start,
                length=length,
                data=(self.data[name][0][start:start + length],
                      self.data[name][1][start:start + length]))
            for name, start, length in list(
                zip(names, starts, sample_lengths))
        ]

    def get_single_segment(self, extract_idx=0, start_value=3000000, sample_length=200000):
        
        #print(sample_length)
        name = list(self.data.keys())[extract_idx]
        sample_length = sample_length or len(self.data[name][0])
        return DataItem(
            name=name,
            start_offset=start_value,
            length=sample_length,
            data=(self.data[name][0][start_value:start_value + sample_length],
                  self.data[name][1][start_value:start_value + sample_length]))

