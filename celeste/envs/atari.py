from absl import flags
from absl import logging
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
    self.train_env = AtariEnv(FLAGS.env_name, num_envs=1, color_mode='gray', device='cpu',
                              rescale=True, frameskip=4, repeat_prob=0.0, clip_rewards=False,
                              episodic_life=False)
    self.train_env.train()

    self.test_env = AtariEnv(FLAGS.env_name, num_envs=1, color_mode='gray', device='cpu',
                             rescale=True, frameskip=4, repeat_prob=0.0, clip_rewards=False,
                             episodic_life=False)

    height, width = self.train_env.screen_shape()
    utils.assert_equal(height, FLAGS.image_height)
    utils.assert_equal(width, FLAGS.image_width)

    pygame.init()
    self.screen = pygame.display.set_mode([width, height], pygame.SCALED, depth=8)

  def quit(self):
    pygame.quit()

  def reset(self):
    if GLOBAL.eval_mode:
      self.next_frame = self.test_env.reset(initial_steps=400).squeeze(-1)
    else:
      self.next_frame = self.train_env.reset(initial_steps=400).squeeze(-1)
    self.next_reward = None
    self.should_end_episode = False

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
    frame = self.next_frame.squeeze(0).cpu().numpy().T
    frame = np.tile(np.expand_dims(frame, -1), [1, 1, 3])
    self.screen.blit(pygame.surfarray.make_surface(frame), (0, 0))
    pygame.display.flip()
    return self.next_frame, None

  def get_inputs_for_frame(self, frame):
    frame = frame.float() / 255.0
    frame = utils.downsample_image_to_input(frame)
    if FLAGS.use_cuda:
      frame = frame.cuda()
    return frame, None

  def get_reward(self):
    return self.next_reward, self.should_end_episode

  def indices_to_actions(self, idxs):
    return idxs

  def indices_to_labels(self, idxs):
    return [_ACTION_NAMES[idxs[i]] for i in range(len(idxs))]

  def end_frame(self, action):
    # speedrun.py only supports single element batches. Add back the batch dim here.
    actions = torch.Tensor([action])
    if GLOBAL.eval_mode:
      observation, reward, should_end_episode, _ = self.test_env.step(actions)
    else:
      observation, reward, should_end_episode, _ = self.train_env.step(actions)
    self.next_frame = observation.squeeze(-1)
    self.next_reward = float(torch.sum(reward).cpu().numpy())
    self.should_end_episode = should_end_episode.all()

