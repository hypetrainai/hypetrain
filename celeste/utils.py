from absl import flags
import importlib
import matplotlib.collections as mcoll
import matplotlib.pyplot as plt
import numpy as np
import pylibtas
import torch
from torch.nn import functional as F

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


def outputs_to_log_probs(outputs):
  if FLAGS.probs_fn == 'softmax':
    return outputs - torch.logsumexp(outputs, 1, keepdim=True)
  elif FLAGS.probs_fn == 'square':
    return torch.log(outputs**2 / torch.sum(outputs**2, 1, keepdim=True))
  else:
    raise ValueError('Invalid probs_fn %s' % FLAGS.probs_fn)


def sample_log_softmax(log_softmax):
  if GLOBAL.eval_mode:
    return np.argmax(log_softmax.numpy(), axis=1)
  elif (FLAGS.random_action_prob > 0 and
        np.random.random() <= FLAGS.random_action_prob):
    N, n_classes = log_softmax.shape
    return np.random.randint(n_classes, size=N)
  else:
    dist = torch.distributions.categorical.Categorical(probs=torch.exp(log_softmax))
    return dist.sample().numpy()


def generate_gaussian_heat_map(image_shape, y, x, sigma=10, amplitude=1.0):
  H, W = image_shape
  y_range = np.arange(0, H)
  x_range = np.arange(0, W)
  x_grid, y_grid = np.meshgrid(x_range, y_range)

  result = amplitude * np.exp((-(y_grid - y)**2 + -(x_grid - x)**2) / (2 * sigma**2))
  return result.astype(np.float32)


def downsample_image_to_input(image):
  if FLAGS.image_height != FLAGS.input_height or FLAGS.image_width != FLAGS.input_width:
    assert FLAGS.image_height % FLAGS.input_height == 0
    assert FLAGS.image_width % FLAGS.input_width == 0
    assert FLAGS.image_width * FLAGS.input_height == FLAGS.image_height * FLAGS.input_width
    image = F.interpolate(image, size=(FLAGS.input_height, FLAGS.input_width), mode='nearest')
  return image


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


def imshow(image, ax=None):
  """Shows the image taking care of any transposes needed."""
  if image.shape[-1] != FLAGS.image_channels:
    image = image.transpose([1, 2, 0])
  assert image.shape[-1] == FLAGS.image_channels, image.shape
  if FLAGS.image_channels == 1:
    image = image.squeeze(-1)
  ax = ax or plt.gca()
  ax.imshow(image)


def plot_trajectory(trajectory, bg=None, ax=None):
  """Plots trajectory list of (y, x) coordinates over bg."""
  ax = ax or plt.gca()
  if bg is not None:
    imshow(bg, ax)
  y, x = zip(*trajectory)
  colorline(x, y, ax=ax, cmap='autumn')


def get_grads(parameters):
  return [None if p.grad is None else p.grad.detach().cpu().numpy().flatten()
          for p in parameters]


def grad_norm(grads, old_grads=None):
  if old_grads is None:
    old_grads = [None] * len(grads)
  diffs = []
  for p, old_p in zip(grads, old_grads):
    if p is None:
      continue
    diffs.append(p - (0 if old_p is None else old_p))
  return np.linalg.norm(np.concatenate(diffs), 2)
