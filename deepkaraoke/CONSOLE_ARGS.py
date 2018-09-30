import argparse
import pprint

def run_from_ipython():
    try:
        __IPYTHON__
        return True
    except NameError:
        return False

parser = argparse.ArgumentParser()

parser.add_argument('--log_dir', type=str, default='trained_models/test_new', help='where to save the models')
parser.add_argument('--module_name', type=str, default='fftmodel')
parser.add_argument('--model_name', type=str, default='Simple')
parser.add_argument('--lr', type=int, default=0.01)

if run_from_ipython():
    ARGS = parser.parse_args([])
else:
    ARGS = parser.parse_args()

pprint.pprint(vars(ARGS))
