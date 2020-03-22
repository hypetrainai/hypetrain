from absl import app
from absl import flags
from absl import logging
logging.set_verbosity(logging.INFO)
import cProfile
import functools
import inspect
import matplotlib
matplotlib.rcParams['axes.formatter.useoffset'] = False
import matplotlib.pyplot as plt
import numpy as np
import os
import pprint
import queue
import subprocess
import time
from tensorboardX import SummaryWriter
import torch
from torch import nn
from torch import optim

from GLOBALS import GLOBAL
import env
import model
import submodules
import utils

flags.DEFINE_string('save_config', '', 'File to save the config for the current run into. Can be loaded using --flagfile.')

flags.DEFINE_string('env', 'envs.celeste.Env', 'class for environment')
flags.DEFINE_string('env_name', 'PongNoFrameskip-v4', 'environment name (for envs.atari.Env)')
flags.DEFINE_string('actor', 'model.ResNetIm2Value', 'class for actor network')
flags.DEFINE_string('critic', 'model.ResNetIm2Value', 'class for critic network')
flags.DEFINE_string('logdir', None, 'logdir')
flags.DEFINE_string('pretrained_model_path', '', 'pretrained model path')
flags.DEFINE_string('pretrained_suffix', 'latest', 'if latest, will load most recent save in dir')
flags.DEFINE_boolean('use_cuda', True, 'Use cuda')
flags.DEFINE_boolean('profile', False, 'Profile code')
flags.DEFINE_boolean('debug', False, 'Debug mode')
flags.DEFINE_boolean('visualize', False, 'Enable visualizations')

flags.DEFINE_integer('batch_size', 1, 'batch size')
flags.DEFINE_integer('image_height', 540, 'image height')
flags.DEFINE_integer('image_width', 960, 'image width')
flags.DEFINE_integer('image_channels', 3, 'image width')
flags.DEFINE_integer('input_height', 270, 'image height')
flags.DEFINE_integer('input_width', 480, 'image height')

flags.DEFINE_integer('max_episodes', 100000, 'stop after this many episodes')
flags.DEFINE_integer('save_every', 100, 'every X number of steps save a model')
flags.DEFINE_integer('eval_every', 100, 'eval every X steps')
flags.DEFINE_boolean('evaluate', False, 'if true, run a single step of eval and exit')

flags.DEFINE_string('movie_file', 'movie.ltm', 'if not empty string, load libTAS input movie file')
flags.DEFINE_string('save_file', 'level1_screen0', 'if not empty string, use save file.')
flags.DEFINE_string('savestate_path', '', 'where to put savestates')
flags.DEFINE_integer('goal_y', 0, 'override goal y coordinate')
flags.DEFINE_integer('goal_x', 0, 'override goal x coordinate')

flags.DEFINE_float('lr', 0.0005, 'learning rate')
flags.DEFINE_float('actor_start_delay', 0, 'delay training of the actor for this many episodes')
flags.DEFINE_float('actor_loss_weight', 1.0, 'weight for actor loss')
flags.DEFINE_float('value_loss_weight', 1.0, 'weight for value loss')
flags.DEFINE_float('entropy_loss_weight', 0.0001, 'weight for entropy loss')
flags.DEFINE_boolean('differential_reward', True, 'Do we use differential rewards?')
flags.DEFINE_float('reward_decay_multiplier', 0.95, 'reward time decay multiplier')
flags.DEFINE_integer('episode_length', 400, 'episode length')
flags.DEFINE_integer('unroll_steps', 1, 'number of steps before each bprop run')
flags.DEFINE_integer('context_frames', 30, 'number of frames passed to the network')
flags.DEFINE_integer('bellman_lookahead_frames', 10, 'number of frames to consider for bellman rollout')
flags.DEFINE_float('clip_grad_norm', 1000.0, 'value to clip gradient norm to.')
flags.DEFINE_float('clip_grad_value', 0.0, 'value to clip gradients to.')
flags.DEFINE_integer('hold_buttons_for', 4, 'hold all buttons for at least this number of frames')
flags.DEFINE_boolean('multitask', True, 'do we use the same network for both A and C?')
flags.DEFINE_string('probs_fn', 'softmax', 'function to convert outputs to probs. softmax or softplus')

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
    if FLAGS.multitask:
      self.actor = utils.import_class(FLAGS.actor)(
          frame_channels, extra_channels, out_dim=[1, self.env.num_actions()],
          output_probs=[False, True])
    else:
      self.actor = utils.import_class(FLAGS.actor)(
          frame_channels, extra_channels, out_dim=self.env.num_actions())
      if not FLAGS.evaluate:
        self.critic = utils.import_class(FLAGS.critic)(
            frame_channels, extra_channels, out_dim=1, output_probs=False)
    if FLAGS.use_cuda:
      self.actor = self.actor.cuda()
      if hasattr(self, 'critic'):
        self.critic = self.critic.cuda()

    self.all_parameters = list(self.actor.parameters())
    optimizer_fn = functools.partial(optim.Adam, lr=FLAGS.lr, amsgrad=True)
    self.optimizer_actor = optimizer_fn(list(self.actor.parameters()))
    if hasattr(self, 'critic'):
      self.all_parameters += list(self.critic.parameters())
      self.all_parameters = list(set(self.all_parameters))
      self.optimizer_critic = optimizer_fn(list(self.critic.parameters()))

    self.processed_frames = 0

    if FLAGS.pretrained_model_path:
      logging.info('Loading pretrained model from %s' % FLAGS.pretrained_model_path)
      state = torch.load(os.path.join(FLAGS.pretrained_model_path, 'train/model_%s.tar' % FLAGS.pretrained_suffix))
      GLOBAL.episode_number = state['episode_number']
      self.actor.load_state_dict(state['actor'])
      self.optimizer_actor.load_state_dict(state['optimizer_actor'])
      if hasattr(self, 'critic'):
        self.critic.load_state_dict(state['critic'])
        self.optimizer_critic.load_state_dict(state['optimizer_critic'])
      logging.info('Done!')

  def quit(self):
    self.env.quit()

  def eval(self):
    self.actor.eval()
    if hasattr(self, 'critic'):
      self.critic.eval()
    GLOBAL.eval_mode = True

  def train(self):
    self.actor.train()
    if hasattr(self, 'critic'):
      self.critic.train()
    GLOBAL.eval_mode = False

  def savestate(self, index):
    logging.info('Saving state %d!' % index)
    self.env.savestate(index)
    self.actor.savestate(index)
    if hasattr(self, 'critic'):
      self.critic.savestate(index)
    self.saved_states[index] = self.frame_buffer[-1:]

  def loadstate(self, index):
    logging.info('Loading state %d!' % index)
    self.env.loadstate(index)
    self.actor.loadstate(index)
    if hasattr(self, 'critic'):
      self.critic.loadstate(index)
    self.frame_buffer = self.saved_states[index].copy()

  def _start_episode(self):
    if GLOBAL.episode_number % FLAGS.save_every == 0 and not GLOBAL.eval_mode:
      state = {
          'episode_number': GLOBAL.episode_number,
          'actor': self.actor.state_dict(),
          'optimizer_actor': self.optimizer_actor.state_dict(),
      }
      if hasattr(self, 'critic'):
        state['critic'] = self.critic.state_dict()
        state['optimizer_critic'] = self.optimizer_critic.state_dict()
      savefile = os.path.join(FLAGS.logdir, 'train/model_%d.tar' % GLOBAL.episode_number)
      torch.save(state, savefile)
      torch.save(state, os.path.join(FLAGS.logdir, 'train/model_latest.tar'))
      logging.info('Saved %s', savefile)

    logging.info('%s episode %d',
        'Evaluating' if GLOBAL.eval_mode else 'Starting',
        GLOBAL.episode_number)
    self.actor.reset()
    if hasattr(self, 'critic'):
      self.critic.reset()
    self.env.reset()
    self.initial_parameters = [p.data.detach().cpu().numpy().flatten() for p in self.all_parameters]

    assert FLAGS.episode_length >= FLAGS.unroll_steps

    self.frame_buffer = []
    self.next_frame_to_process = 0
    self.weights = np.zeros((FLAGS.episode_length, FLAGS.batch_size), dtype=np.bool)
    self.rewards = np.zeros((FLAGS.episode_length, FLAGS.batch_size), dtype=np.float32)
    self.log_softmaxes = [None] * FLAGS.episode_length
    self.sampled_idx = np.zeros((FLAGS.episode_length, FLAGS.batch_size), dtype=np.int64)
    self.Rs = np.zeros((FLAGS.episode_length + 1, FLAGS.batch_size), dtype=np.float32)
    self.Vs = np.zeros((FLAGS.episode_length + 1, FLAGS.batch_size), dtype=np.float32)
    self.As = np.zeros((FLAGS.episode_length, FLAGS.batch_size), dtype=np.float32)
    self.value_losses = np.zeros((FLAGS.episode_length, FLAGS.batch_size), dtype=np.float32)
    self.actor_losses = np.zeros((FLAGS.episode_length, FLAGS.batch_size), dtype=np.float32)
    self.entropy_losses = np.zeros((FLAGS.episode_length, FLAGS.batch_size), dtype=np.float32)
    if FLAGS.use_cuda:
      torch.cuda.synchronize()
      torch.cuda.empty_cache()
    self.episode_start_time = time.time()

  def _bprop(self):
    assert self.processed_frames > 0

    self.optimizer_actor.zero_grad()
    if hasattr(self, 'critic'):
      self.optimizer_critic.zero_grad()

    weight_tensor = utils.to_tensor(self.weights[self.next_frame_to_process:self.processed_frames].astype(np.float32))
    rewards_tensor = utils.to_tensor(self.rewards[self.next_frame_to_process:self.processed_frames])
    sampled_idx_tensor = utils.to_tensor(self.sampled_idx[self.next_frame_to_process:self.processed_frames])

    if FLAGS.multitask:
      final_V = self.actor.forward(self.processed_frames)[0].detach().cpu().numpy()
    elif hasattr(self, 'critic'):
      final_V = self.critic.forward(self.processed_frames).detach().cpu().numpy()
    else:
      final_V = [0] * FLAGS.batch_size
    self.Vs[self.processed_frames] = np.reshape(final_V, [FLAGS.batch_size])
    # Bootstrap off V for sequences that haven't terminated.
    R = self.Vs[self.processed_frames]
    # Weight == 0 means the sequence terminated so set extra reward to 0.
    R[self.weights[self.processed_frames - 1] == 0] = 0
    self.Rs[self.processed_frames] = R
    R = utils.to_tensor(R)

    for i in reversed(range(self.next_frame_to_process, self.processed_frames)):
      ii = i - self.next_frame_to_process
      outputs = self.actor.forward(i)
      if FLAGS.multitask:
        V, log_softmax = outputs
      else:
        if hasattr(self, 'critic'):
          V = self.critic.forward(i)
        else:
          V = utils.to_tensor([0])
        log_softmax = outputs
      V = torch.reshape(V, [FLAGS.batch_size])

      R = FLAGS.reward_decay_multiplier * R + rewards_tensor[ii]
      if FLAGS.bellman_lookahead_frames == 0:
        A = R - V
      else:
        blf = min(i + FLAGS.bellman_lookahead_frames, self.processed_frames)
        R_blf = (FLAGS.reward_decay_multiplier**blf) * utils.to_tensor(self.Rs[blf])
        V_blf = (FLAGS.reward_decay_multiplier**blf) * utils.to_tensor(self.Vs[blf])
        A = R - R_blf + V_blf - V

      value_loss = A**2
      self.value_losses[i] = value_loss.detach().cpu().numpy()

      R = R.detach()
      V = V.detach()
      A = A.detach()

      self.Rs[i] = R.cpu().numpy()
      self.Vs[i] = V.cpu().numpy()
      self.As[i] = A.cpu().numpy()

      assert torch.equal(self.log_softmaxes[i], log_softmax)
      log_action_probs = log_softmax.gather(1, sampled_idx_tensor[ii].unsqueeze(-1))
      log_action_probs = torch.squeeze(log_action_probs, -1)
      actor_loss = -log_action_probs * A
      self.actor_losses[i] = actor_loss.detach().cpu().numpy()
      entropy = torch.sum(-torch.exp(log_softmax) * log_softmax, dim=-1)
      # Maximize entropy -> trend toward uniform distribution.
      entropy_loss = -entropy
      self.entropy_losses[i] = entropy_loss.detach().cpu().numpy()
      if not GLOBAL.eval_mode:
        if FLAGS.debug:
          grads = utils.get_grads(self.all_parameters)
        batch_weight = torch.max(torch.sum(weight_tensor[ii]), utils.to_tensor([1.0]))
        for name, loss in [
            ('actor_loss', FLAGS.actor_loss_weight * actor_loss),
            ('value_loss', FLAGS.value_loss_weight * value_loss),
            ('entropy_loss', FLAGS.entropy_loss_weight * entropy_loss),
        ]:
          loss = torch.sum(weight_tensor[ii] * loss) / batch_weight
          loss.backward(retain_graph=True)
          if FLAGS.debug:
            new_grads = utils.get_grads(self.all_parameters)
            grad_norm = utils.grad_norm(new_grads, grads)
            utils.add_summary('scalar', name + '_grad_norm/%d' % i, grad_norm)
            grads = new_grads

      if (i + 1) % FLAGS.action_summary_frames == 0:
        self.env.add_action_summaries(
            i, self.frame_buffer, log_softmax.detach().cpu().numpy(), self.sampled_idx[i])

    if not GLOBAL.eval_mode:
      assert not (FLAGS.clip_grad_value and FLAGS.clip_grad_norm)
      if FLAGS.unroll_steps > 1 and (FLAGS.clip_grad_value or FLAGS.clip_grad_norm):
        raise ValueError('There is a bug with unroll_steps > 1 and grad clipping, '
                         'since grad clipping occurs on the accumulated gradients.')
      if FLAGS.clip_grad_value:
        nn.utils.clip_grad_value_(self.actor.parameters(), FLAGS.clip_grad_value)
        if hasattr(self, 'critic'):
          nn.utils.clip_grad_value_(self.critic.parameters(), FLAGS.clip_grad_value)
      elif FLAGS.clip_grad_norm:
        nn.utils.clip_grad_norm_(self.actor.parameters(), FLAGS.clip_grad_norm)
        if hasattr(self, 'critic'):
          nn.utils.clip_grad_norm_(self.critic.parameters(), FLAGS.clip_grad_norm)
      if GLOBAL.episode_number >= FLAGS.actor_start_delay:
        self.optimizer_actor.step()
      if hasattr(self, 'critic'):
        self.optimizer_critic.step()

    self.next_frame_to_process = self.processed_frames

  def _finish_episode(self):
    assert self.processed_frames > 0
    episode_time = time.time() - self.episode_start_time

    self.env.finish_episode(self.processed_frames, self.frame_buffer)

    ep_len = np.sum(self.weights, axis=0)
    avg_episode_length = np.mean(ep_len)
    utils.add_summary('scalar', 'avg_episode_length', avg_episode_length)
    avg_total_reward = np.mean(np.sum(self.weights * self.rewards, axis=0))
    utils.add_summary('scalar', 'avg_total_reward', avg_total_reward)
    avg_reward = avg_total_reward / np.maximum(avg_episode_length, 1.0)
    utils.add_summary('scalar', 'avg_reward', avg_reward)

    fig = plt.figure()
    plt.plot(self.rewards[:ep_len[0], 0])
    utils.add_summary('figure', 'out/reward', fig)
    fig = plt.figure()
    plt.plot(self.Rs[:ep_len[0], 0])
    utils.add_summary('figure', 'out/reward_cumul', fig)
    fig = plt.figure()
    plt.plot(self.Vs[:ep_len[0], 0])
    utils.add_summary('figure', 'out/value', fig)
    fig = plt.figure()
    plt.plot(self.As[:ep_len[0], 0])
    utils.add_summary('figure', 'out/advantage', fig)
    fig = plt.figure()
    all_Vs = []
    all_Rs = []
    for i in range(FLAGS.batch_size):
      all_Vs.extend(self.Vs[:ep_len[i], i].tolist())
      all_Rs.extend(self.Rs[:ep_len[i], i].tolist())
    explained_variance = utils.explained_variance(np.array(all_Vs), np.array(all_Rs))
    utils.add_summary('scalar', 'loss/explained_variance', explained_variance)
    total_frames = np.maximum(np.sum(ep_len), 1.0)
    plt.plot(self.value_losses[:ep_len[0], 0])
    utils.add_summary('figure', 'loss/value', fig)
    avg_value_loss = np.sum(self.weights * self.value_losses) / total_frames
    utils.add_summary('scalar', 'loss/value_avg', avg_value_loss)
    fig = plt.figure()
    plt.plot(self.actor_losses[:ep_len[0], 0])
    utils.add_summary('figure', 'loss/actor', fig)
    avg_actor_loss = np.sum(self.weights * self.actor_losses) / total_frames
    utils.add_summary('scalar', 'loss/actor_avg', avg_actor_loss)
    fig = plt.figure()
    plt.plot(self.entropy_losses[:ep_len[0], 0])
    utils.add_summary('figure', 'loss/entropy', fig)
    avg_entropy_loss = np.sum(self.weights * self.entropy_losses) / total_frames
    utils.add_summary('scalar', 'loss/entropy_avg', avg_entropy_loss)
    logging.info(('=======================\n'
                  'Total Reward: %.3f Reward: %.3f Average length: %.1f Explained variance: %.3f\n'
                  'Value loss: %.3f Actor loss: %.3f Entropy loss: %.3f FPS: %.1f'),
        avg_total_reward, avg_reward, avg_episode_length, explained_variance,
        avg_value_loss, avg_actor_loss, avg_entropy_loss, total_frames / episode_time)

    if not GLOBAL.eval_mode:
      updated_parameters = [p.data.detach().cpu().numpy().flatten() for p in self.all_parameters]
      update_norm = np.linalg.norm([np.linalg.norm(updated - initial)
                                    for updated, initial
                                    in zip(updated_parameters, self.initial_parameters)])
      utils.add_summary('scalar', 'update_norm', update_norm)
      if update_norm < 1e-5:
        logging.warn('Small update norm detected (%.2e). Model may not be training.')

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
      assert frame.shape == (FLAGS.batch_size, FLAGS.image_channels,
                             FLAGS.image_height, FLAGS.image_width), frame.shape
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
      reward, done = self.env.get_reward()
      assert reward.shape == (FLAGS.batch_size,), reward.shape
      assert done.shape == (FLAGS.batch_size,), done.shape
      if self.processed_frames > 1:
        # If the previous frame was done, we propagate it to this frame.
        done = np.maximum(done, 1 - self.weights[self.processed_frames - 2])
      self.weights[self.processed_frames - 1] = done == 0
      self.rewards[self.processed_frames - 1] = reward * (1.0 - done)
      if done.all() or self.processed_frames % FLAGS.unroll_steps == 0:
        self._bprop()
      if done.all() or self.processed_frames >= FLAGS.episode_length:
        return self._finish_episode()

    with torch.no_grad():
      log_softmax = self.actor.forward(self.processed_frames)
      if FLAGS.multitask:
        log_softmax = log_softmax[1]
    self.log_softmaxes[self.processed_frames] = log_softmax
    idxs = utils.sample_log_softmax(log_softmax)
    self.sampled_idx[self.processed_frames] = idxs
    self.processed_frames += 1

    # Predicted idxs include a batch dimension.
    actions = self.env.indices_to_actions(idxs)
    # Returned actions is for next N frames.
    return [actions] * FLAGS.hold_buttons_for

  def Run(self):
    action_queue = queue.Queue()
    while True:
      frame, actions = self.env.start_frame()
      if frame is None:
        break
      if actions:
        action_queue.put(actions)
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

  os.makedirs(FLAGS.logdir, exist_ok=True)
  summary_dir = os.path.join(FLAGS.logdir, 'eval' if FLAGS.evaluate else 'train')
  os.makedirs(summary_dir, exist_ok=True)
  GLOBAL.summary_writer = SummaryWriter(summary_dir)

  key_flags = FLAGS.get_key_flags_for_module(os.path.basename(__file__))
  flags_dict = {flag.name: flag.value for flag in key_flags}
  logging.info('\n%s', pprint.pformat(flags_dict))
  for k, v in flags_dict.items():
    GLOBAL.summary_writer.add_text(k, str(v))

  if FLAGS.save_config:
    with open(FLAGS.save_config, 'w') as f:
      for flag in sorted(key_flags, key=lambda x: x.name):
        if flag.name != 'save_config':
          f.write(flag.serialize() + '\n')

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
  flags.mark_flag_as_required('logdir')
  app.run(main)
