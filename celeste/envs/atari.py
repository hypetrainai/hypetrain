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

    self._train_envs = cmd_util.make_vec_env(FLAGS.env_name, 'atari', FLAGS.batch_size, seed=None)
    self._test_envs = cmd_util.make_vec_env(FLAGS.env_name, 'atari', FLAGS.batch_size, seed=None)

    height, width, _ = self._train_envs.observation_space.shape
    utils.assert_equal(height, FLAGS.image_height)
    utils.assert_equal(width, FLAGS.image_width)

    self._action_meanings = cmd_util.make_env(FLAGS.env_name, 'atari').get_action_meanings()

    if FLAGS.visualize:
      pygame.init()
      tile_height = int(math.sqrt(FLAGS.batch_size))
      tile_width = int(math.ceil(FLAGS.batch_size / tile_height))
      self._screen = pygame.display.set_mode([width * tile_width, height * tile_height], pygame.SCALED, depth=8)

  def quit(self):
    if FLAGS.visualize:
      pygame.quit()

  def reset(self):
    env_to_use = self._test_envs if GLOBAL.eval_mode else self._train_envs
    self._next_frame = env_to_use.reset().reshape(
        FLAGS.batch_size, 1, FLAGS.image_height, FLAGS.image_width)
    self._next_reward = None
    self._done = None

  def frame_channels(self):
    return 1

  def extra_channels(self):
    return 0

  def num_actions(self):
    return self._train_envs.action_space.n

  def start_frame(self):
    if FLAGS.visualize:
      for event in pygame.event.get():
        if event.type == pygame.QUIT:
          return None, None
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
    env_to_use = self._test_envs if GLOBAL.eval_mode else self._train_envs
    observation, reward, done, _ = env_to_use.step(actions)
    self._next_frame = observation.reshape(
        FLAGS.batch_size, 1, FLAGS.image_height, FLAGS.image_width)
    self._next_reward = reward
    self._done = done
