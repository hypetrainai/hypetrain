from absl import app
from absl import flags
from absl import logging
logging.set_verbosity(logging.INFO)
import cProfile
import inspect
import matplotlib
matplotlib.rcParams['axes.formatter.useoffset'] = False
import matplotlib.pyplot as plt
import numpy as np
import os
import pprint
import queue
import subprocess
from tensorboardX import SummaryWriter
import torch
from torch import nn
from torch import optim

from GLOBALS import GLOBAL
import env
import model
import utils

flags.DEFINE_string('save_config', '', 'File to save the config for the current run into. Can be loaded using --flagfile.')

flags.DEFINE_string('env', 'envs.celeste.Env', 'class for environment')
flags.DEFINE_string('env_name', 'PongNoFrameskip-v4', 'environment name (for envs.atari.Env)')
flags.DEFINE_string('actor', 'model.ResNetIm2Value', 'class for actor network')
flags.DEFINE_string('critic', 'model.ResNetIm2Value', 'class for critic network')
flags.DEFINE_string('logdir', 'trained_models/vbellman', 'logdir')
flags.DEFINE_string('pretrained_model_path', '', 'pretrained model path')
flags.DEFINE_string('pretrained_suffix', 'latest', 'if latest, will load most recent save in dir')
flags.DEFINE_boolean('use_cuda', True, 'Use cuda')
flags.DEFINE_boolean('profile', False, 'Profile code')

flags.DEFINE_integer('max_episodes', 100000, 'stop after this many episodes')
flags.DEFINE_integer('save_every', 100, 'every X number of steps save a model')
flags.DEFINE_integer('eval_every', 100, 'eval every X steps')
flags.DEFINE_boolean('evaluate', False, 'if true, run a single step of eval and exit')

flags.DEFINE_string('movie_file', 'movie.ltm', 'if not empty string, load libTAS input movie file')
flags.DEFINE_string('save_file', 'level1_screen0', 'if not empty string, use save file.')
flags.DEFINE_integer('goal_y', 0, 'override goal y coordinate')
flags.DEFINE_integer('goal_x', 0, 'override goal x coordinate')

flags.DEFINE_integer('image_height', 540, 'image height')
flags.DEFINE_integer('image_width', 960, 'image width')
flags.DEFINE_integer('image_channels', 3, 'image width')
flags.DEFINE_integer('input_height', 270, 'image height')
flags.DEFINE_integer('input_width', 480, 'image height')

flags.DEFINE_float('lr', 0.0005, 'learning rate')
flags.DEFINE_float('actor_start_delay', 10, 'delay training of the actor for this many episodes')
flags.DEFINE_float('value_loss_weight', 1.0, 'weight for value loss')
flags.DEFINE_float('entropy_loss_weight', 0.0001, 'weight for entropy loss')
flags.DEFINE_float('reward_scale', 1.0/10.0, 'multiplicative scale for the reward function')
flags.DEFINE_float('reward_decay_multiplier', 0.95, 'reward time decay multiplier')
flags.DEFINE_integer('episode_length', 150, 'episode length')
flags.DEFINE_integer('context_frames', 30, 'number of frames passed to the network')
flags.DEFINE_integer('bellman_lookahead_frames', 10, 'number of frames to consider for bellman rollout')
flags.DEFINE_float('clip_grad_norm', 1000.0, 'value to clip gradient norm to.')
flags.DEFINE_float('clip_grad_value', 0.0, 'value to clip gradients to.')
flags.DEFINE_integer('hold_buttons_for', 4, 'hold all buttons for at least this number of frames')

flags.DEFINE_float('random_goal_prob', 0.0, 'probability that we choose a random goal')
flags.DEFINE_float('random_action_prob', 0.1, 'probability that we choose a random action')
flags.DEFINE_float('random_savestate_prob', 1.0/1000.0, 'probability that we savestate each frame')
flags.DEFINE_float('random_loadstate_prob', 0.3, 'probability that we load a custom state each episode')
flags.DEFINE_float('num_custom_savestates', 3, 'number of custom savestates, not including the default one')
flags.DEFINE_integer('action_summary_frames', 50, 'number of frames between action summaries')

FLAGS = flags.FLAGS


class Trainer(object):

  def __init__(self):
    self.env = utils.import_class(FLAGS.env)()
    self.env.reset()

    self.saved_states = {}
    self.frame_buffer = []

    frame_channels = self.env.frame_channels()
    extra_channels = self.env.extra_channels()
    num_actions = self.env.num_actions()
    self.actor = utils.import_class(FLAGS.actor)(frame_channels, extra_channels, out_dim=num_actions)
    self.critic = utils.import_class(FLAGS.critic)(frame_channels, extra_channels, out_dim=1, use_softmax=False)
    if FLAGS.use_cuda:
      self.actor = self.actor.cuda()
      self.critic = self.critic.cuda()
    self.optimizer_actor = optim.Adam(list(self.actor.parameters()), lr=FLAGS.lr)
    self.optimizer_critic = optim.Adam(list(self.critic.parameters()), lr=FLAGS.lr)

    self.processed_frames = 0

    if FLAGS.pretrained_model_path:
      logging.info('Loading pretrained model from %s' % FLAGS.pretrained_model_path)
      state = torch.load(os.path.join(FLAGS.pretrained_model_path, 'train/model_%s.tar' % FLAGS.pretrained_suffix))
      GLOBAL.episode_number = state['episode_number']
      self.actor.load_state_dict(state['actor'])
      self.optimizer_actor.load_state_dict(state['optimizer_actor'])
      self.critic.load_state_dict(state['critic'])
      self.optimizer_critic.load_state_dict(state['optimizer_critic'])
      logging.info('Done!')

  def quit(self):
    self.env.quit()

  def eval(self):
    self.actor.eval()
    self.critic.eval()
    GLOBAL.eval_mode = True

  def train(self):
    self.actor.train()
    self.critic.train()
    GLOBAL.eval_mode = False

  def savestate(self, index):
    logging.info('Saving state %d!' % index)
    self.env.savestate(index)
    self.actor.savestate(index)
    self.critic.savestate(index)
    self.saved_states[index] = self.frame_buffer[-1:]

  def loadstate(self, index):
    logging.info('Loading state %d!' % index)
    self.env.loadstate(index)
    self.actor.loadstate(index)
    self.critic.loadstate(index)
    self.frame_buffer = self.saved_states[index].copy()

  def _start_episode(self):
    if GLOBAL.episode_number % FLAGS.save_every == 0 and not GLOBAL.eval_mode:
      state = {
          'episode_number': GLOBAL.episode_number,
          'actor': self.actor.state_dict(),
          'optimizer_actor': self.optimizer_actor.state_dict(),
          'critic': self.critic.state_dict(),
          'optimizer_critic': self.optimizer_critic.state_dict()
      }
      savefile = os.path.join(FLAGS.logdir, 'train/model_%d.tar' % GLOBAL.episode_number)
      torch.save(state, savefile)
      torch.save(state, os.path.join(FLAGS.logdir, 'train/model_latest.tar'))
      logging.info('Saved %s', savefile)

    logging.info('%s episode %d',
        'Evaluating' if GLOBAL.eval_mode else 'Starting',
        GLOBAL.episode_number)
    self.actor.reset()
    self.critic.reset()
    self.env.reset()
    self.frame_buffer = []
    self.rewards = []
    self.softmaxes = []
    self.sampled_idx = []
    if FLAGS.use_cuda:
      torch.cuda.empty_cache()

  def _finish_episode(self):
    assert self.processed_frames > 0
    self.env.finish_episode(self.processed_frames, self.frame_buffer)

    utils.assert_equal(len(self.rewards), self.processed_frames)
    utils.assert_equal(len(self.softmaxes), self.processed_frames)
    utils.assert_equal(len(self.sampled_idx), self.processed_frames)

    utils.add_summary('scalar', 'episode_length', self.processed_frames)
    utils.add_summary('scalar', 'avg_reward', sum(self.rewards) / self.processed_frames)

    R = self.rewards[self.processed_frames - 1]
    # reward = (1 + gamma + gamma^2 + ...) * reward
    R *= 1.0 / (1.0 - FLAGS.reward_decay_multiplier)
    Rs = [R]
    final_V = self.critic.forward(self.processed_frames).view([]).detach().cpu().numpy()
    Vs = [final_V]
    As = [0]
    value_losses = []
    actor_losses = []
    entropy_losses = []
    self.optimizer_actor.zero_grad()
    self.optimizer_critic.zero_grad()
    for i in reversed(range(self.processed_frames)):
      R = FLAGS.reward_decay_multiplier * R + self.rewards[i]
      V = self.critic.forward(i).view([])

      blf = min(FLAGS.bellman_lookahead_frames, self.processed_frames - i)
      assert blf > 0
      V_bellman = (R - (FLAGS.reward_decay_multiplier**blf) * Rs[-blf]
                   + (FLAGS.reward_decay_multiplier**blf) * Vs[-blf])
      A = V_bellman - V

      value_loss = FLAGS.value_loss_weight * A**2
      value_losses.append(value_loss.detach().cpu().numpy())
      if not GLOBAL.eval_mode:
        value_loss.backward(retain_graph=True)

      Rs.append(R)
      Vs.append(V.detach().cpu().numpy())
      As.append(A.detach().cpu().numpy())

      softmax = self.actor.forward(i)
      assert torch.eq(self.softmaxes[i], softmax.cpu()).all()
      entropy = -torch.sum(softmax * torch.log(softmax))
      actor_loss = -torch.log(softmax[0, self.sampled_idx[i][0]]) * A
      actor_losses.append(actor_loss.detach().cpu().numpy())
      entropy_loss = FLAGS.entropy_loss_weight / (entropy + 1e-6)
      entropy_losses.append(entropy_loss.detach().cpu().numpy())
      if not GLOBAL.eval_mode:
        (actor_loss + entropy_loss).backward(retain_graph=True)

      if (i + 1) % FLAGS.action_summary_frames == 0:
        self.env.add_action_summaries(
            i, self.frame_buffer, softmax[0].detach().cpu().numpy(), self.sampled_idx[i][0])

    if not GLOBAL.eval_mode:
      utils.add_summary('scalar', 'actor_grad_norm', utils.grad_norm(self.actor))
      utils.add_summary('scalar', 'critic_grad_norm', utils.grad_norm(self.critic))
      assert not (FLAGS.clip_grad_value and FLAGS.clip_grad_norm)
      if FLAGS.clip_grad_value:
        nn.utils.clip_grad_value_(self.actor.parameters(), FLAGS.clip_grad_value)
        nn.utils.clip_grad_value_(self.critic.parameters(), FLAGS.clip_grad_value)
      elif FLAGS.clip_grad_norm:
        nn.utils.clip_grad_norm_(self.actor.parameters(), FLAGS.clip_grad_norm)
        nn.utils.clip_grad_norm_(self.critic.parameters(), FLAGS.clip_grad_norm)
      if GLOBAL.episode_number >= FLAGS.actor_start_delay:
        self.optimizer_actor.step()
      self.optimizer_critic.step()

    fig = plt.figure()
    plt.plot(self.rewards)
    utils.add_summary('figure', 'out/reward', fig)
    fig = plt.figure()
    plt.plot(list(reversed(Rs[1:])))
    utils.add_summary('figure', 'out/reward_cumul', fig)
    fig = plt.figure()
    plt.plot(list(reversed(Vs[1:])))
    utils.add_summary('figure', 'out/value', fig)
    fig = plt.figure()
    plt.plot(list(reversed(As[1:])))
    utils.add_summary('figure', 'out/advantage', fig)
    fig = plt.figure()
    plt.plot(list(reversed(value_losses)))
    utils.add_summary('figure', 'loss/value', fig)
    fig = plt.figure()
    plt.plot(list(reversed(actor_losses)))
    utils.add_summary('figure', 'loss/actor', fig)
    fig = plt.figure()
    plt.plot(list(reversed(entropy_losses)))
    utils.add_summary('figure', 'loss/entropy', fig)

    # Start next episode.
    GLOBAL.episode_number += 1
    if GLOBAL.eval_mode:
      GLOBAL.episode_number -= 1  # Don't increase episode number for eval.
      self.train()
    elif GLOBAL.episode_number % FLAGS.eval_every == 0:
      self.eval()

    if GLOBAL.episode_number >= FLAGS.max_episodes:
      return None

    self.processed_frames = 0
    return self.process_frame(frame=None)

  def process_frame(self, frame):
    """Returns a list of button inputs for the next N frames."""
    if self.processed_frames == 0:
      self._start_episode()

    if frame is None:
      # Starting new episode, perform a loadstate.
      if np.random.uniform() < FLAGS.random_loadstate_prob and not GLOBAL.eval_mode:
        custom_savestates = set(self.saved_states.keys())
        if len(custom_savestates) > 1:
          custom_savestates.remove(0)
        self.loadstate(int(np.random.choice(list(custom_savestates))))
      else:
        self.loadstate(0)
    else:
      assert frame.shape == (FLAGS.image_channels, FLAGS.image_height, FLAGS.image_width), frame.shape
      if isinstance(frame, torch.Tensor):
        frame_np = frame.clone().cpu().numpy()
      else:
        frame_np = frame
      self.frame_buffer.append(frame_np)
      inputs = self.env.get_inputs_for_frame(frame)
      self.actor.set_inputs(self.processed_frames, *inputs)
      if hasattr(self, 'critic'):
        self.critic.set_inputs(self.processed_frames, *inputs)

      if not self.saved_states:
        assert self.env.can_savestate()
        # First (default) savestate.
        self.savestate(0)
      elif (FLAGS.num_custom_savestates
            and self.env.can_savestate()
            and np.random.uniform() < FLAGS.random_savestate_prob):
        self.savestate(1 + np.random.randint(FLAGS.num_custom_savestates))

    if self.processed_frames > 0:
      reward, should_end_episode = self.env.get_reward()
      assert isinstance(reward, float), type(reward)
      self.rewards.append(reward * FLAGS.reward_scale)
      if should_end_episode or self.processed_frames >= FLAGS.episode_length:
        return self._finish_episode()

    with torch.no_grad():
      softmax = self.actor.forward(self.processed_frames).detach().cpu()
    self.softmaxes.append(softmax)
    idxs = utils.sample_softmax(softmax)
    self.sampled_idx.append(idxs)
    self.processed_frames += 1

    # Predicted idxs include a batch dimension.
    actions = self.env.indices_to_actions(idxs)
    # Currently we have only 1 env = batch_size 1.
    action = actions[0]
    # Returned actions is for next N frames.
    return [action] * FLAGS.hold_buttons_for

  def Run(self):
    action_queue = queue.Queue()
    while True:
      frame, action = self.env.start_frame()
      if frame is None:
        break
      if action:
        action_queue.put(action)
      if action_queue.empty():
        predicted_actions = self.process_frame(frame)
        if not predicted_actions:
          break
        assert isinstance(predicted_actions, (list, tuple))
        for predicted_action in predicted_actions:
          action_queue.put(predicted_action)
      self.env.end_frame(action_queue.get())


def main(argv):
  del argv  # unused

  key_flags = FLAGS.get_key_flags_for_module(os.path.basename(__file__))
  logging.info('\n%s', pprint.pformat({flag.name: flag.value for flag in key_flags}))

  if FLAGS.save_config:
    with open(FLAGS.save_config, 'w') as f:
      for flag in sorted(key_flags, key=lambda x: x.name):
        if flag.name != 'save_config':
          f.write(flag.serialize() + '\n')

  os.makedirs(FLAGS.logdir, exist_ok=True)
  summary_dir = os.path.join(FLAGS.logdir, 'eval' if FLAGS.evaluate else 'train')
  os.makedirs(summary_dir, exist_ok=True)
  GLOBAL.summary_writer = SummaryWriter(summary_dir)

  tensorboard = subprocess.Popen(['tensorboard', '--logdir', os.path.abspath(FLAGS.logdir)],
                                 stderr=subprocess.DEVNULL)

  GLOBAL.eval_mode = False
  GLOBAL.episode_number = 0

  try:
    trainer = Trainer()
    if FLAGS.evaluate:
      assert FLAGS.pretrained_model_path, 'evaluate requires pretrained_model_path.'
      FLAGS.max_episodes = 0
      trainer.eval()
    if FLAGS.profile:
      FLAGS.max_episodes = 0
      cProfile.run('trainer.Run()')
    else:
      trainer.Run()
  finally:
    GLOBAL.summary_writer.close()
    if tensorboard:
      tensorboard.terminate()
    trainer.quit()


if __name__ == '__main__':
  app.run(main)
