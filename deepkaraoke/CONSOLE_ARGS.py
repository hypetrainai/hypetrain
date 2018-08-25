import argparse

def run_from_ipython():
    try:
        __IPYTHON__
        return True
    except NameError:
        return False

parser = argparse.ArgumentParser()

parser.add_argument('--log_dir', type=str, default='trained_models/test', help='where to save the models')
parser.add_argument('--module_name', type=str, default='fftmodel')
parser.add_argument('--model_name', type=str, default='Simple')

if run_from_ipython():
    ARGS = parser.parse_args([])
else:
    ARGS = parser.parse_args()