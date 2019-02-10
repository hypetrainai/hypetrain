import argparse
import os
import pprint
import sys
from tensorboardX import SummaryWriter
from types import ModuleType
import traceback
import warnings


class _AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(_AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def _run_from_ipython():
    try:
        __IPYTHON__
        return True
    except NameError:
        return False


def _warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    log = file if hasattr(file,'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))
warnings.showwarning = _warn_with_traceback


# Hack to be able to access properties on the module.
class _ModuleWrapper(ModuleType):

    def __init__(self, *args, **kwargs):
        super(_ModuleWrapper, self).__init__(*args, **kwargs)
        self.GLOBAL = _AttrDict()

        parser = argparse.ArgumentParser()

        parser.add_argument('--log_dir', type=str, default='trained_models/firstrun_largedataset', help='where to save the models')
        parser.add_argument('--resume', type=bool, default=True, help='resume training from checkpoint')
        parser.add_argument('--checkpoint', type=str, default='latest', help='suffix for checkpoint file to restore from')
        parser.add_argument('--data_dir', type=str, default='data/16k_LARGE_fixed', help='the directory that contains train/test directories.')
        parser.add_argument('--module_name', type=str, default='fftmodel')
        parser.add_argument('--model_name', type=str, default='GeneratorDeepSupervision')
        parser.add_argument('--lr', type=float, default=0.02)
        parser.add_argument('--clip_grad_norm', type=float, default=-1)
        parser.add_argument('--batch_size', type=int, default=24)
        parser.add_argument('--train_seqlen', type=int, default=10000, help='sequence length during training')
        parser.add_argument('--max_steps', type=int, default=1000000, help='number of iterations to train for')

        parser.add_argument('--sample_rate', type=int, default=16000)
        parser.add_argument('--hop_length_ms', type=int, default=10)
        parser.add_argument('--window_length_ms', type=int, default=40)
        parser.add_argument('--n_mels', type=int, default=128)
        parser.add_argument('--fmin', type=int, default=0)

        parser.add_argument('--debug', type=bool, default=True, help='Boolean switch for controlling debug code.')

        #TENSORBOARD ARGS
        parser.add_argument('--image_summaries', type=bool, default=False, help='Enable image summaries on tensorboard?')

        self.parser = parser
        self._FLAGS = None

    @property
    def FLAGS(self):
        if not self._FLAGS:
            if _run_from_ipython():
                FLAGS = self.parser.parse_args([])
            else:
                FLAGS = self.parser.parse_args()
            pprint.pprint(vars(FLAGS))

            os.makedirs(FLAGS.log_dir, exist_ok=True)
            self.GLOBAL.summary_writer = SummaryWriter(FLAGS.log_dir)
            self._FLAGS = FLAGS
        return self._FLAGS

sys.modules[__name__] = _ModuleWrapper(__name__)
