import argparse
import pprint

def run_from_ipython():
    try:
        __IPYTHON__
        return True
    except NameError:
        return False

parser = argparse.ArgumentParser()

parser.add_argument('--log_dir', type=str, default='trained_models/withnorm_highlr_fromscratch', help='where to save the models')
parser.add_argument('--module_name', type=str, default='fftmodel')
parser.add_argument('--model_name', type=str, default='Generator')
parser.add_argument('--lr', type=float, default=0.01)

parser.add_argument('--sample_rate', type=int, default=16000)
parser.add_argument('--hop_length_ms', type=int, default=10)
parser.add_argument('--window_length_ms', type=int, default=40)
parser.add_argument('--n_mels', type=int, default=128)
parser.add_argument('--fmin', type=int, default=0)

#TENSORBOARD ARGS
parser.add_argument('--image_summaries', type=bool, default=True, help='Enable image summaries on tensorboard?')


if run_from_ipython():
    ARGS = parser.parse_args([])
else:
    ARGS = parser.parse_args()

pprint.pprint(vars(ARGS))
