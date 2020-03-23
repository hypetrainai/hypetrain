from absl import flags
from absl import logging
from baselines.common import cmd_util
import math
import numpy as np
import os
import pygame
import signal
import torch
from torch.nn import functional as F

from GLOBALS import GLOBAL
import env
import utils

FLAGS = flags.FLAGS


class Env(env.Env):

  def __init__(self):
    super(Env, self).__init__()

    wrapper_kwargs = dict(episode_life=False)
    self._envs = [cmd_util.make_env(FLAGS.env_name, 'atari', wrapper_kwargs=wrapper_kwargs)
                  for _ in range(FLAGS.batch_size)]

    self._action_meanings = self._envs[0].get_action_meanings()
    height, width, _ = self._envs[0].observation_space.shape
    utils.assert_equal(height, FLAGS.image_height)
    utils.assert_equal(width, FLAGS.image_width)

    if FLAGS.visualize:
      pygame.init()
      tile_height = int(math.sqrt(FLAGS.batch_size))
      tile_width = int(math.ceil(FLAGS.batch_size / tile_height))
      self._screen = pygame.display.set_mode([width * tile_width, height * tile_height], pygame.SCALED, depth=8)

  def quit(self):
    if FLAGS.visualize:
      pygame.quit()

  def reset(self):
    self._next_frame = np.stack([e.reset() for e in self._envs]).reshape(
        FLAGS.batch_size, 1, FLAGS.image_height, FLAGS.image_width)
    self._next_reward = None
    self._done = None

  def frame_channels(self):
    return 1

  def extra_channels(self):
    return 0

  def num_actions(self):
    return self._envs[0].action_space.n

  def start_frame(self):
    if FLAGS.visualize:
      wait = False
      while wait:
        for event in pygame.event.get():
          if event.type == pygame.QUIT:
            return None, None
          if event.type == pygame.KEYDOWN:
            wait=False
      frame = self._next_frame
      # [batch, channel, height, width] -> [batch, width, height, channel].
      frame = np.tile(np.transpose(frame, [0, 3, 2, 1]), [1, 1, 1, 3])
      for i in range(len(frame)):
        tile_height = int(math.sqrt(FLAGS.batch_size))
        tile_width = int(math.ceil(FLAGS.batch_size / tile_height))
        start_x = (i % tile_width) * FLAGS.image_width
        start_y = (i // tile_width) * FLAGS.image_height
        self._screen.blit(pygame.surfarray.make_surface(frame[i]), (start_x, start_y))
      pygame.display.flip()

    input_frame = utils.to_tensor(self._next_frame.astype(np.float32) / 255.0)
    scripted_actions = None
    return input_frame, scripted_actions

  def get_inputs_for_frame(self, frame):
    extra_channels = None
    return utils.downsample_image_to_input(frame), extra_channels

  def get_reward(self):
    return self._next_reward, self._done

  def indices_to_actions(self, idxs):
    return idxs

  def indices_to_labels(self, idxs):
    return [self._action_meanings[idxs[i]] for i in range(len(idxs))]

  def end_frame(self, actions):
    observation, reward, done, _ = zip(*[e.step(a) for e, a in zip(self._envs, actions)])
    self._next_frame = np.stack(observation).reshape(
        FLAGS.batch_size, 1, FLAGS.image_height, FLAGS.image_width)
    self._next_reward = np.stack(reward)
    self._done = np.stack(done)
