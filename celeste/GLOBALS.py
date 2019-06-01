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

        parser.add_argument('--log_dir', type=str, default='trained_models/test', help='where to save the models')

        parser.add_argument('--movie_file', type=str, default='movie.ltm', help='if not empty string, load libTAS input movie file')
        parser.add_argument('--save_file', type=str, default='level1_screen4', help='if not empty string, use save file.')

        parser.add_argument('--interactive', type=bool, default=True, help='interactive mode (enter buttons on command line)')

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
