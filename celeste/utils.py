import matplotlib.collections as mcoll
import matplotlib.pyplot as plt
import numpy as np
import pylibtas
import torch

from absl import flags
FLAGS = flags.FLAGS

def assert_equal(a, b):
  assert a == b, (a, b)


button_dict = {
    'a': pylibtas.SingleInput.IT_CONTROLLER1_BUTTON_A,
    'b': pylibtas.SingleInput.IT_CONTROLLER1_BUTTON_B,
#    'x': pylibtas.SingleInput.IT_CONTROLLER1_BUTTON_X,
#    'y': pylibtas.SingleInput.IT_CONTROLLER1_BUTTON_Y,
    'rt': pylibtas.SingleInput.IT_CONTROLLER1_BUTTON_RIGHTSHOULDER,
#    'lt': pylibtas.SingleInput.IT_CONTROLLER1_BUTTON_LEFTSHOULDER,
    'u': pylibtas.SingleInput.IT_CONTROLLER1_BUTTON_DPAD_UP,
    'd': pylibtas.SingleInput.IT_CONTROLLER1_BUTTON_DPAD_DOWN,
    'l': pylibtas.SingleInput.IT_CONTROLLER1_BUTTON_DPAD_LEFT,
    'r': pylibtas.SingleInput.IT_CONTROLLER1_BUTTON_DPAD_RIGHT,
}


def class2button(key):
  if not class2button.dict:
    action_button_dict = {
      0: [''],
      1: ['a'],
      2: ['b'],
      3: ['rt'],
      4: ['a','b'],
      5: ['a','rt'],
      6: ['rt', 'b'],
      7: ['a','b','rt']
    }
    dpad_button_dict = {
      0: [''],
      1: ['r'],
      2: ['l'],
      3: ['u'],
      4: ['d'],
      5: ['r','u'],
      6: ['u','l'],
      7: ['l','d'],
      8: ['d','r']
    }
    for action_key in action_button_dict:
      for dpad_key in dpad_button_dict:
        final_key = action_key * len(dpad_button_dict) + dpad_key
        final_value = action_button_dict[action_key] + dpad_button_dict[dpad_key]
        final_value = [key for key in final_value if key]
        class2button.dict[final_key] = final_value
  return class2button.dict[key]
class2button.dict = {}
class2button(0)


def sample_action(softmax, greedy=False):
    
  N, n_classes = softmax.shape
    
  if greedy:
    sample = np.argmax(softmax.numpy(), axis=1)
  else:
    sample = torch.distributions.categorical.Categorical(probs=softmax).sample().numpy()
  
  if FLAGS.random_action_prob > 0:
    if np.random.random() <= FLAGS.random_action_prob:
        sample = np.random.randint(0, n_classes, (N))
        
  sample_mapped = [class2button(sample[i]) for i in range(len(sample))]
  return sample, sample_mapped


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

