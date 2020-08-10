from absl import flags
from absl import logging
from gym.envs.classic_control import rendering
import numpy as np
import os
from pyglet.gl import *
import sys
import torch
from torch.nn import functional as F

from GLOBALS import GLOBAL
import env
import utils

FLAGS = flags.FLAGS


def angle_normalize(x):
  return (((x+np.pi) % (2*np.pi)) - np.pi)


class Env(env.Env):

  def __init__(self):
    super().__init__()
    self.max_speed = 8
    self.max_torque = 2.
    self.dt = .05
    self.g = 10.
    self.m = 1.
    self.l = 1.
    self.viewer = None

  def reset(self):
    self.theta = np.random.uniform(low=-np.pi, high=np.pi, size=FLAGS.batch_size)
    self.thetadot = np.random.uniform(low=-1, high=1, size=FLAGS.batch_size)

  def quit(self):
    if self.viewer:
      self.viewer.close()
      self.viewer = None
    super().quit()

  def frame_channels(self):
    return 3

  def extra_channels(self):
    return 0

  def num_actions(self):
    return 256

  def start_frame(self):
    if FLAGS.visualize:
      if self.viewer is None:
        self.viewer = rendering.Viewer(500, 500)
        self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
        rod = rendering.make_capsule(1, .2)
        self.pole_transform = rendering.Transform()
        rod.add_attr(self.pole_transform)
        self.viewer.add_geom(rod)
        axle = rendering.make_circle(.05)
        axle.set_color(0, 0, 0)
        self.viewer.add_geom(axle)

      self.pole_transform.set_rotation(self.theta[0] + np.pi / 2)
      self.viewer.render()

    frames = np.zeros([FLAGS.batch_size, 3, 1, 1])
    scripted_actions = None
    return frames, scripted_actions

  def get_inputs_for_frame(self, unused_frame):
    input_frame = np.array(list(zip(np.cos(self.theta), np.sin(self.theta), self.thetadot)))
    input_frame = np.reshape(input_frame, [FLAGS.batch_size, 3, 1, 1])
    input_frame = utils.to_tensor(input_frame.astype(np.float32))
    extra_channels = None
    return input_frame, extra_channels

  def indices_to_labels(self, idxs):
    """Given softmax indices, return a batch of string action names."""
    space = np.linspace(-self.max_torque, self.max_torque, self.num_actions())
    return [space[i] for i in idxs]

  def get_reward(self):
    costs = angle_normalize(self.theta) ** 2 + .1 * self.thetadot ** 2 + .001 * (self.u ** 2)
    done = np.array([False] * FLAGS.batch_size)
    return -costs, done

  def end_frame(self, actions):
    th = self.theta
    thdot = self.thetadot
    g = self.g
    m = self.m
    l = self.l
    dt = self.dt

    space = np.linspace(-self.max_torque, self.max_torque, self.num_actions())
    self.u = np.array([space[i] for i in actions])

    self.thetadot = thdot + (-3 * g / (2 * l) * np.sin(th + np.pi) + 3. / (m * l ** 2) * self.u) * dt
    self.theta = th + self.thetadot * dt
    self.thetadot = np.clip(self.thetadot, -self.max_speed, self.max_speed)
