import matplotlib.collections as mcoll
import matplotlib.pyplot as plt
import numpy as np

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
                final_key = action_key * 9 + dpad_key
                if dpad_button_dict[dpad_key][0] == '':
                    final_value = action_button_dict[action_key]
                elif action_button_dict[action_key][0] == '':
                    final_value = dpad_button_dict[dpad_key]
                else:
                    final_value = action_button_dict[action_key] + dpad_button_dict[dpad_key]
                class2button.dict[final_key] = final_value
    return class2button.dict[key]
class2button.dict = {}


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


def plotTrajectory(bg, trajectory, ax=None):
    """Plots trajectory list of (y, x) coordinates over bg."""
    ax = ax or plt.gca()
    ax.imshow(np.transpose(bg, [1, 2, 0]))
    y, x = zip(*trajectory)
    colorline(x, y, ax=ax, cmap='autumn')
