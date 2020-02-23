from absl import flags
from absl import logging
import math
import numpy as np
import os
import pygame
import signal
import torch
from torch.nn import functional as F
from torchcule.atari import Env as AtariEnv

from GLOBALS import GLOBAL
import env
import utils

FLAGS = flags.FLAGS

_ACTION_NAMES = [
    'NOOP', 'RIGHT', 'LEFT', 'DOWN', 'UP', 'FIRE', 'UPRIGHT',
    'UPLEFT', 'DOWNRIGHT', 'DOWNLEFT', 'UPFIRE', 'RIGHTFIRE',
    'LEFTFIRE', 'DOWNFIRE', 'UPRIGHTFIRE', 'UPLEFTFIRE',
    'DOWNRIGHTFIRE', 'DOWNLEFTFIRE']


class Env(env.Env):

  def __init__(self):
    super(Env, self).__init__()

    # See https://github.com/NVlabs/cule/blob/master/examples/utils/launcher.py for description.
    self.train_env = AtariEnv(FLAGS.env_name, num_envs=FLAGS.batch_size, color_mode='gray', device='cpu',
                              rescale=True, frameskip=4, repeat_prob=0.0, clip_rewards=False,
                              episodic_life=False)
    self.train_env.train()

    self.test_env = AtariEnv(FLAGS.env_name, num_envs=FLAGS.batch_size, color_mode='gray', device='cpu',
                             rescale=True, frameskip=4, repeat_prob=0.0, clip_rewards=False,
                             episodic_life=False)

    height, width = self.train_env.screen_shape()
    utils.assert_equal(height, FLAGS.image_height)
    utils.assert_equal(width, FLAGS.image_width)

    pygame.init()
    tile_height = int(math.sqrt(FLAGS.batch_size))
    tile_width = int(math.ceil(FLAGS.batch_size / tile_height))
    self.screen = pygame.display.set_mode([width * tile_width, height * tile_height], pygame.SCALED, depth=8)

  def quit(self):
    pygame.quit()

  def reset(self):
    if GLOBAL.eval_mode:
      self.next_frame = self.test_env.reset(initial_steps=400).squeeze(-1).unsqueeze(1)
    else:
      self.next_frame = self.train_env.reset(initial_steps=400).squeeze(-1).unsqueeze(1)
    self.next_reward = None
    self.done = None

  def frame_channels(self):
    return 1

  def extra_channels(self):
    return 0

  def num_actions(self):
    return self.train_env.minimal_actions().size(0)

  def start_frame(self):
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        return None, None
    frame = self.next_frame.cpu().numpy()
    # [batch, channel, height, width] -> [batch, width, height, channel].
    frame = np.tile(np.transpose(frame, [0, 3, 2, 1]), [1, 1, 1, 3])
    for i in range(len(frame)):
      tile_height = int(math.sqrt(FLAGS.batch_size))
      tile_width = int(math.ceil(FLAGS.batch_size / tile_height))
      start_x = (i % tile_width) * FLAGS.image_width
      start_y = (i // tile_width) * FLAGS.image_height
      self.screen.blit(pygame.surfarray.make_surface(frame[i]), (start_x, start_y))
    pygame.display.flip()
    return self.next_frame, None

  def get_inputs_for_frame(self, frame):
    frame = frame.float() / 255.0
    frame = utils.downsample_image_to_input(frame)
    if FLAGS.use_cuda:
      frame = frame.cuda()
    return frame, None

  def get_reward(self):
    return self.next_reward, self.done

  def indices_to_actions(self, idxs):
    return idxs

  def indices_to_labels(self, idxs):
    return [_ACTION_NAMES[idxs[i]] for i in range(len(idxs))]

  def end_frame(self, actions):
    actions = torch.Tensor(actions)
    if GLOBAL.eval_mode:
      observation, reward, done, _ = self.test_env.step(actions)
    else:
      observation, reward, done, _ = self.train_env.step(actions)
    self.next_frame = observation.squeeze(-1).unsqueeze(1)
    self.next_reward = reward.cpu().numpy()
    self.done = done.cpu().numpy()
