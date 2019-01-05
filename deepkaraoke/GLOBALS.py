import argparse
import os
import pprint
from tensorboardX import SummaryWriter


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

GLOBAL = AttrDict()


def run_from_ipython():
    try:
        __IPYTHON__
        return True
    except NameError:
        return False


parser = argparse.ArgumentParser()

parser.add_argument('--log_dir', type=str, default='trained_models/deepsupervision_firstattempt', help='where to save the models')
parser.add_argument('--module_name', type=str, default='fftmodel')
parser.add_argument('--model_name', type=str, default='GeneratorDeepSupervision')
parser.add_argument('--lr', type=float, default=0.002)
parser.add_argument('--batch_size', type=int, default=24)
parser.add_argument('--train_seqlen', type=int, default=10000, help='sequence length during training')

parser.add_argument('--sample_rate', type=int, default=16000)
parser.add_argument('--hop_length_ms', type=int, default=10)
parser.add_argument('--window_length_ms', type=int, default=40)
parser.add_argument('--n_mels', type=int, default=128)
parser.add_argument('--fmin', type=int, default=0)

parser.add_argument('--debug', type=bool, default=True, help='Boolean switch for controlling debug code.')

#TENSORBOARD ARGS
parser.add_argument('--image_summaries', type=bool, default=False, help='Enable image summaries on tensorboard?')


if run_from_ipython():
    FLAGS = parser.parse_args([])
else:
    FLAGS = parser.parse_args()
pprint.pprint(vars(FLAGS))

os.makedirs(FLAGS.log_dir, exist_ok=True)
GLOBAL.summary_writer = SummaryWriter(FLAGS.log_dir)

