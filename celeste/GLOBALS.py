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

        parser.add_argument('--pretrained_model_path', type=str, default='', help='pretrained model path')
        parser.add_argument('--pretrained_suffix', type=str, default='latest', help='if latest, will load most recent save in dir')

        parser.add_argument('--log_dir', type=str, default='trained_models/firstmodel_bellman', help='where to save the models')

        parser.add_argument('--save_every', type=int, default=100, help='every X number of steps save a model')

        parser.add_argument('--movie_file', type=str, default='movie.ltm', help='if not empty string, load libTAS input movie file')
        parser.add_argument('--save_file', type=str, default='level1_screen4', help='if not empty string, use save file.')

        parser.add_argument('--interactive', type=bool, default=False, help='interactive mode (enter buttons on command line)')


        #actual model arguments now
        parser.add_argument('--image_height', type=int, default=540)
        parser.add_argument('--image_width', type=int, default=960)
        parser.add_argument('--image_channels', type=int, default=3)

        parser.add_argument('--num_actions', type=int, default=72)

        parser.add_argument('--lr', type=float, default=0.0002)
        parser.add_argument('--entropy_weight', type=float, default=0.01)
        parser.add_argument('--reward_decay_multiplier', type=int, default=0.95, help='reward function decay multiplier')
        parser.add_argument('--episode_length', type=int, default=200, help='episode length')
        parser.add_argument('--context_frames', type=int, default=30, help='number of frames passed to the network')

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
            train_dir = os.path.join(FLAGS.log_dir, 'train')
            os.makedirs(train_dir, exist_ok=True)
            self.GLOBAL.summary_writer = SummaryWriter(train_dir)
            self._FLAGS = FLAGS
        return self._FLAGS

sys.modules[__name__] = _ModuleWrapper(__name__)
