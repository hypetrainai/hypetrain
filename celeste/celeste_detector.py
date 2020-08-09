from absl import flags
from absl import logging
import numpy as np

FLAGS = flags.FLAGS


color_dict = {
  'red_hair' : np.array([179, 60, 69]),
  'blue_hair' : np.array([74, 183, 255]),
  'blue_hair_2' : np.array([99, 152, 218]),
  'blue_hair_3' : np.array([81, 174, 252]),
  'brown' : np.array([142, 64, 54]),
  'blue_shirt' : np.array([99, 123, 255]),
  'skin' : np.array([244, 215, 180])
}


class CelesteDetector:

  def __init__(self, threshold=10, search_delta_with_prior=100):
    self.thres = threshold
    self.sdwp = search_delta_with_prior
    if self.sdwp % 2 == 1:
      self.sdwp += 1
    self.saved_states = {}
    self.death_clock_limit = int(25 / FLAGS.hold_buttons_for)
    self.death_clock = 0

  def savestate(self, index):
    self.saved_states[index] = self.death_clock

  def loadstate(self, index):
    self.death_clock = self.saved_states[index]

  def detect(self, im, prior_coord=None):
    # state 0 is red hair
    # state 1 is blue hair
    # state 2 is other (probably white hair)
    # state 3 is death clock initiated
    # state -1 is no detection
    y = None
    x = None
    state = -1
    col = color_dict['blue_shirt']

    mask = self._get_mask(im, col, prior_coord=prior_coord)
    if np.sum(mask) == 0 and prior_coord is not None:
      mask = self._get_mask(im, col, prior_coord=None)
      if np.sum(mask) != 0:
        prior_coord = None
    if np.sum(mask) == 0:
      self.death_clock += 1
      if self.death_clock < self.death_clock_limit and prior_coord is not None:
        y, x, state = prior_coord[0], prior_coord[1], 3
    else:
      self.death_clock = 0
      y, x = self._COM_coordinates(mask)
      if prior_coord is not None:
        miny = np.maximum(0, prior_coord[0] - self.sdwp//2)
        minx = np.maximum(0, prior_coord[1] - self.sdwp//2)
        y += miny
        x += minx
      state = self._get_state(im, prior_coord)

    if state != -1:
      logging.debug('Character Location: (%d, %d), State: %d' % (y, x, state))
    else:
      logging.debug('Character Location: Not Found!')
    return y, x, state

  def _get_state(self, im, prior_coord):
    if np.sum(self._get_mask(im, color_dict['red_hair'], prior_coord)) > 0:
      return 0
    if np.sum(self._get_mask(im, color_dict['blue_hair'], prior_coord)) > 0:
      return 1
    if np.sum(self._get_mask(im, color_dict['blue_hair_2'], prior_coord)) > 0:
      return 1
    if np.sum(self._get_mask(im, color_dict['blue_hair_3'], prior_coord)) > 0:
      return 1
    return 2

  def _get_mask(self, im, col, prior_coord):
    col = col.reshape([3, 1, 1])
    if prior_coord is None:
      im_col = np.sum((im - col)**2, 0)
    else:
      miny = np.maximum(0, prior_coord[0] - self.sdwp//2)
      minx = np.maximum(0, prior_coord[1] - self.sdwp//2)
      im_sub = im[:, miny:prior_coord[0]+self.sdwp//2, minx:prior_coord[1]+self.sdwp//2]
      im_col = np.sum((im_sub - col)**2, 0)
    return im_col < self.thres**2

  def _COM_coordinates(self, mask):
    y, x = np.where(mask)
    return [np.around(np.mean(y)).astype(np.int32),
            np.around(np.mean(x)).astype(np.int32)]

