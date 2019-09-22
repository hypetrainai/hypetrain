from absl import flags
from absl import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import signal
import torch
from torch.nn import functional as F
from torchcule.atari import Env as AtariEnv

from GLOBALS import GLOBAL
import env
import utils

FLAGS = flags.FLAGS


class Env(env.Env):

  def __init__(self):
    super(Env, self).__init__()

    # See https://github.com/NVlabs/cule/blob/master/examples/utils/launcher.py for description.
    self.train_env = AtariEnv(FLAGS.env_name, num_envs=1, color_mode='gray', device='gpu',
                              rescale=True, frameskip=4, repeat_prob=0.0, clip_rewards=False,
                              episodic_life=False)
    self.train_env.train()

    self.test_env = AtariEnv(FLAGS.env_name, num_envs=10, color_mode='gray', device='cpu',
                             rescale=True, frameskip=4, repeat_prob=0.0, clip_rewards=False,
                             episodic_life=False)

    height, width = self.train_env.screen_shape()
    utils.assert_equal(height, FLAGS.image_height)
    utils.assert_equal(width, FLAGS.image_height)

  def reset(self):
    if GLOBAL.eval_mode:
      self.next_frame = self.test_env.reset(initial_steps=400).squeeze(-1)
    else:
      self.next_frame = self.train_env.reset(initial_steps=400).squeeze(-1)
    print(self.next_frame.shape)
    self.next_reward = None
    self.should_end_episode = False

    plt.close('all')
    plt.ion()
    fig = plt.figure()
    self.visualization = fig.imshow(self.next_frame.numpy(), animated=True)
    fig.axis('off')
    plt.tight_layout()
    plt.show()

  def frame_channels(self):
    return 1

  def extra_channels(self):
    return 0

  def num_actions(self):
    return self.train_env.minimal_actions.size(0)

  def start_frame(self):
    self.visualization.set_data(self.next_frame)
    return self.next_frame, None

  def get_inputs_for_frame(self, frame):
    frame = utils.downsample_image_to_input(frame)
    return frame, None

  def get_reward(self):
    return self.next_reward, self.should_end_episode

  def index_to_action(self, idx):
    return idx

  def end_frame(self, action):
    if GLOBAL.eval_mode:
      observation, reward, should_end_episode, _ = eval_env.step(action)
    else:
      observation, reward, should_end_episode, _ = train_env.step(action)
    print(observation.shape)
    print(reward.shape)
    print(should_end_episode.shape)
    self.next_frame = observation.squeeze(-1)
    self.next_reward = reward
    self.should_end_episode = should_end_episode

