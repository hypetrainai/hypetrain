from absl import flags
import importlib
import matplotlib.collections as mcoll
import matplotlib.pyplot as plt
import numpy as np
import pylibtas
import torch

from GLOBALS import GLOBAL

FLAGS = flags.FLAGS


def assert_equal(a, b):
  assert a == b, (a, b)


def import_class(path):
  module, class_name = path.rsplit('.', 1)
  return getattr(importlib.import_module(module), class_name)


def add_summary(summary_type, name, value, **kwargs):
  summary_prefix = 'eval/' if GLOBAL.eval_mode else ''
  summary_fn = getattr(GLOBAL.summary_writer, 'add_%s' % summary_type)
  summary_fn(summary_prefix + name, value, GLOBAL.episode_number, **kwargs)


def sample_softmax(softmax):
  if GLOBAL.eval_mode:
    return np.argmax(softmax.numpy(), axis=1)
  elif (FLAGS.random_action_prob > 0 and
        np.random.random() <= FLAGS.random_action_prob):
    N, n_classes = softmax.shape
    return np.random.randint(0, n_classes, (N))
  else:
    return torch.distributions.categorical.Categorical(probs=softmax).sample().numpy()
  return sample


def generate_gaussian_heat_map(image_shape, y, x, sigma=10, amplitude=1.0):
  H, W = image_shape
  y_range = np.arange(0, H)
  x_range = np.arange(0, W)
  x_grid, y_grid = np.meshgrid(x_range, y_range)

  result = amplitude * np.exp((-(y_grid - y)**2 + -(x_grid - x)**2) / (2 * sigma**2))
  return result.astype(np.float32)


def colorline(x, y, z=None, ax=None, cmap='copper', norm=plt.Normalize(0.0, 1.0),
              linewidth=3, alpha=1.0):
  """
  http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
  http://matplotlib.org/examples/pylab_examples/multicolored_line.html
  Plot a colored line with coordinates x and y
  Optionally specify colors in the array z
  Optionally specify a colormap, a norm function and a line width
  """
  # Default colors equally spaced on [0,1]:
  if z is None:
    z = np.linspace(0.0, 1.0, len(x))

  # Special case if a single number:
  # to check for numerical input -- this is a hack
  if not hasattr(z, "__iter__"):
    z = np.array([z])

  z = np.asarray(z)

  points = np.array([x, y]).T.reshape(-1, 1, 2)
  segments = np.concatenate([points[:-1], points[1:]], axis=1)
  lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm,
                            linewidth=linewidth, alpha=alpha)

  ax = ax or plt.gca()
  ax.add_collection(lc)
  return lc


def plot_trajectory(bg, trajectory, ax=None):
  """Plots trajectory list of (y, x) coordinates over bg."""
  ax = ax or plt.gca()
  ax.imshow(np.transpose(bg, [1, 2, 0]))
  y, x = zip(*trajectory)
  colorline(x, y, ax=ax, cmap='autumn')


def grad_norm(network):
  total_norm = 0
  for p in network.parameters():
    param_norm = p.grad.data.norm(2)
    total_norm += param_norm.item()**2
  return total_norm**0.5

