from absl import app
from absl import flags
from absl import logging
logging.set_verbosity(logging.INFO)
import cProfile
import matplotlib
matplotlib.rcParams['axes.formatter.useoffset'] = False
import matplotlib.pyplot as plt
import numpy as np
import os
import pprint
import pylibtas
import queue
import signal
import subprocess
from tensorboardX import SummaryWriter
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim

from GLOBALS import GLOBAL
import celeste_detector
import environment
import model
import utils


flags.DEFINE_string('actor_network', 'SimpleLSTMModel', 'class for actor network')
flags.DEFINE_string('critic_network', 'SimpleLSTMModel', 'class for critic network')
flags.DEFINE_string('pretrained_model_path', '', 'pretrained model path')
flags.DEFINE_string('pretrained_suffix', 'latest', 'if latest, will load most recent save in dir')
flags.DEFINE_string('logdir', 'trained_models/lstmtest', 'logdir')
flags.DEFINE_boolean('use_cuda', True, 'Use cuda')
flags.DEFINE_boolean('profile', False, 'Profile code')

flags.DEFINE_integer('max_episodes', 100000, 'stop after this many episodes')
flags.DEFINE_integer('save_every', 100, 'every X number of steps save a model')

flags.DEFINE_string('movie_file', 'movie.ltm', 'if not empty string, load libTAS input movie file')
flags.DEFINE_string('save_file', 'level1_screen4', 'if not empty string, use save file.')
flags.DEFINE_integer('goal_y', 107, 'goal pixel coordinate in y')
flags.DEFINE_integer('goal_x', 611, 'goal pixel coordinate in x')
#flags.DEFINE_integer('goal_y', 481, 'goal pixel coordinate in y')
#flags.DEFINE_integer('goal_x', 604, 'goal pixel coordinate in x')

flags.DEFINE_boolean('interactive', False, 'interactive mode (enter buttons on command line)')

flags.DEFINE_integer('image_height', 540, 'image height')
flags.DEFINE_integer('image_width', 960, 'image width')
flags.DEFINE_integer('input_height', 270, 'image height')
flags.DEFINE_integer('input_width', 480, 'image height')

flags.DEFINE_float('lr', 0.0005, 'learning rate')
flags.DEFINE_float('actor_start_delay', 10, 'delay training of the actor for this many episodes')
flags.DEFINE_float('entropy_weight', 0.0001, 'weight for entropy loss')
flags.DEFINE_float('reward_scale', 1.0/10.0, 'multiplicative scale for the reward function')
flags.DEFINE_float('reward_decay_multiplier', 0.95, 'reward time decay multiplier')
flags.DEFINE_integer('episode_length', 150, 'episode length')
flags.DEFINE_integer('context_frames', 30, 'number of frames passed to the network')
flags.DEFINE_integer('bellman_lookahead_frames', 5, 'number of frames to consider for bellman rollout')
flags.DEFINE_float('clip_grad_norm', 1000.0, 'value to clip gradient norm to.')
flags.DEFINE_float('clip_grad_value', 0.0, 'value to clip gradients to.')
flags.DEFINE_integer('hold_buttons_for', 4, 'hold all buttons for at least this number of frames')

flags.DEFINE_float('random_goal_prob', 0.0, 'probability that we choose a random goal')
flags.DEFINE_float('random_savestate_prob', 1.0/1000.0, 'probability that we savestate each frame')
flags.DEFINE_float('random_loadstate_prob', 0.3, 'probability that we load a custom state each episode')
flags.DEFINE_float('num_custom_savestates', 3, 'number of custom savestates, not including the default one')
flags.DEFINE_integer('action_summary_frames', 50, 'number of frames between action summaries')

FLAGS = flags.FLAGS


button_dict = {
    'a': pylibtas.SingleInput.IT_CONTROLLER1_BUTTON_A,
    'b': pylibtas.SingleInput.IT_CONTROLLER1_BUTTON_B,
#    'x': pylibtas.SingleInput.IT_CONTROLLER1_BUTTON_X,
#    'y': pylibtas.SingleInput.IT_CONTROLLER1_BUTTON_Y,
    'rt': pylibtas.SingleInput.IT_CONTROLLER1_BUTTON_RIGHTSHOULDER,
#    'lt': pylibtas.SingleInput.IT_CONTROLLER1_BUTTON_LEFTSHOULDER,
    'u': pylibtas.SingleInput.IT_CONTROLLER1_BUTTON_DPAD_UP,
    'd': pylibtas.SingleInput.IT_CONTROLLER1_BUTTON_DPAD_DOWN,
    'l': pylibtas.SingleInput.IT_CONTROLLER1_BUTTON_DPAD_LEFT,
    'r': pylibtas.SingleInput.IT_CONTROLLER1_BUTTON_DPAD_RIGHT,
}


def sample_action(softmax):
    sample = torch.distributions.categorical.Categorical(probs=softmax).sample().numpy()
    sample_mapped = [utils.class2button(sample[i]) for i in range(len(sample))]
    return sample, sample_mapped


def generate_gaussian_heat_map(image_shape, y, x, sigma=10, amplitude=1.0):
    H, W = image_shape
    y_range = np.arange(0, H)
    x_range = np.arange(0, W)
    x_grid, y_grid = np.meshgrid(x_range, y_range)

    result = amplitude * np.exp((-(y_grid - y)**2 + -(x_grid - x)**2) / (2 * sigma**2))
    return result.astype(np.float32)


class FrameProcessor(object):

  def __init__(self, env):
    self.env = env
    self.det = celeste_detector.CelesteDetector()
    self.saved_states = {}

    self.episode_number = 0
    self.processed_frames = 0

    frame_channels = 4
    extra_channels = 1 + len(button_dict)
    actor_network = getattr(model, FLAGS.actor_network)
    self.actor = actor_network(frame_channels, extra_channels, out_dim=len(utils.class2button.dict))
    critic_network = getattr(model, FLAGS.critic_network)
    self.critic = critic_network(frame_channels, extra_channels, out_dim=1, use_softmax=False)
    if FLAGS.use_cuda:
      self.actor = self.actor.cuda()
      self.critic = self.critic.cuda()
    self.optimizer_actor = optim.Adam(list(self.actor.parameters()), lr=FLAGS.lr)
    self.optimizer_critic = optim.Adam(list(self.critic.parameters()), lr=FLAGS.lr)

    if FLAGS.pretrained_model_path:
      logging.info('Loading pretrained model from %s' % FLAGS.pretrained_model_path)
      self.actor.load_state_dict(torch.load(
        FLAGS.pretrained_model_path + '/celeste_model_actor_%s.pt' % FLAGS.pretrained_suffix))
      self.critic.load_state_dict(torch.load(
        FLAGS.pretrained_model_path + '/celeste_model_critic_%s.pt' % FLAGS.pretrained_suffix))
      logging.info('Done!')

  def savestate(self, index):
    logging.info('Saving state %d!' % index)
    self.env.savestate(index)
    self.det.savestate(index)
    self.actor.savestate(index)
    self.critic.savestate(index)
    self.saved_states[index] = (self.trajectory[-1:], self.frame_buffer[-1:])

  def loadstate(self, index):
    logging.info('Loading state %d!' % index)
    self.env.loadstate(index)
    self.det.loadstate(index)
    self.actor.loadstate(index)
    self.critic.loadstate(index)
    trajectory, frame_buffer = self.saved_states[index]
    self.trajectory = trajectory.copy()
    self.frame_buffer = frame_buffer.copy()

  def _generate_goal_state(self):
    if np.random.uniform() < FLAGS.random_goal_prob:
      self.goal_y = np.random.randint(50, FLAGS.image_height - 50)
      self.goal_x = np.random.randint(50, FLAGS.image_width - 50)
    else:
      self.goal_y = FLAGS.goal_y
      self.goal_x = FLAGS.goal_x

  def _set_inputs_from_frame(self, frame):
    y, x, state = self.det.detect(frame, prior_coord=self.trajectory[-1] if self.trajectory else None)
    self.trajectory.append((y, x))

    window_shape = [FLAGS.image_height, FLAGS.image_width]
    # generate the full frame input by concatenating gaussian heat maps.
    if state == -1:
      gaussian_current_position = np.zeros(window_shape, dtype=np.float32)
    else:
      gaussian_current_position = generate_gaussian_heat_map(window_shape, y, x)

    frame = frame.astype(np.float32).transpose([2, 0, 1]) / 255.0
    self.frame_buffer.append(frame)
    input_frame = torch.cat([torch.tensor(frame), torch.tensor(gaussian_current_position).unsqueeze(0)], 0)

    gaussian_goal_position = generate_gaussian_heat_map(window_shape, self.goal_y, self.goal_x)

    last_frame_buttons = torch.zeros([len(button_dict)] + window_shape)
    if self.sampled_action:
      pressed = utils.class2button(self.sampled_action[-1][0])
      for i, button in enumerate(button_dict.keys()):
        if button in pressed:
          last_frame_buttons[i] = 1.0

    extra_channels = torch.cat([torch.tensor(gaussian_goal_position).unsqueeze(0), last_frame_buttons], 0)

    if FLAGS.image_height != FLAGS.input_height or FLAGS.image_width != FLAGS.input_width:
      assert FLAGS.image_height % FLAGS.input_height == 0
      assert FLAGS.image_width % FLAGS.input_width == 0
      assert FLAGS.image_width * FLAGS.input_height == FLAGS.image_height * FLAGS.input_width
      input_frame = F.interpolate(input_frame.unsqueeze(0), size=(FLAGS.input_height, FLAGS.input_width), mode='nearest').squeeze(0)
      extra_channels = F.interpolate(extra_channels.unsqueeze(0), size=(FLAGS.input_height, FLAGS.input_width), mode='nearest').squeeze(0)

    if FLAGS.use_cuda:
      input_frame = input_frame.cuda()
      extra_channels = extra_channels.cuda()

    self.actor.set_inputs(self.processed_frames, input_frame, extra_channels)
    self.critic.set_inputs(self.processed_frames, input_frame, extra_channels)

  def _reward_for_current_state(self):
    """Returns (rewards, should_end_episode) given state."""
    reward = 0
    should_end_episode = False
    y, x = self.trajectory[-1]

    if y is None:
      # Assume death
      reward -= 3
      should_end_episode = True
      y, x = self.trajectory[-2]
    dist_to_goal = np.sqrt((y - self.goal_y)**2 + (x - self.goal_x)**2)
    reward += 50 - 10 * dist_to_goal**0.33
    if not should_end_episode:
      if self.processed_frames >= FLAGS.episode_length:
        should_end_episode = True
    return reward * FLAGS.reward_scale, should_end_episode

  def _finish_episode(self):
    assert self.processed_frames > 0
    utils.assert_equal(len(self.trajectory), self.processed_frames + 1)
    utils.assert_equal(len(self.frame_buffer), self.processed_frames + 1)
    utils.assert_equal(len(self.rewards), self.processed_frames)
    utils.assert_equal(len(self.softmaxes), self.processed_frames)
    utils.assert_equal(len(self.sampled_action), self.processed_frames)

    GLOBAL.summary_writer.add_scalar('episode_length', self.processed_frames, self.episode_number)
    GLOBAL.summary_writer.add_scalar('final_reward', self.rewards[-1], self.episode_number)
    GLOBAL.summary_writer.add_scalar('best_reward', max(self.rewards), self.episode_number)
    # GLOBAL.summary_writer.add_video('input_frames', self.frame_buffer, self.episode_number, fps=60)

    fig = plt.figure()
    plt.scatter(self.goal_x, self.goal_y, facecolors='none', edgecolors='r')
    utils.plot_trajectory(self.frame_buffer[-1], self.trajectory)
    GLOBAL.summary_writer.add_figure('trajectory', fig, self.episode_number)

    R = self.rewards[self.processed_frames - 1]
    # reward = (1 + gamma + gamma^2 + ...) * reward
    R *= 1.0 / (1.0 - FLAGS.reward_decay_multiplier)
    Rs = [R]
    final_V = self.critic.forward(self.processed_frames).view([]).detach().cpu().numpy()
    Vs = [final_V]
    As = [0]
    actor_losses = []
    entropy_losses = []
    self.optimizer_actor.zero_grad()
    self.optimizer_critic.zero_grad()
    for i in reversed(range(self.processed_frames)):
      R = FLAGS.reward_decay_multiplier * R + self.rewards[i]
      V = self.critic.forward(i).view([])
      ((R - V)**2).backward(retain_graph=i != 0)
      V = V.detach()

      blf = min(FLAGS.bellman_lookahead_frames, self.processed_frames - i)
      assert blf > 0
      V_bellman = (R - (FLAGS.reward_decay_multiplier**blf) * Rs[-blf]
                   + (FLAGS.reward_decay_multiplier**blf) * Vs[-blf])
      A = V_bellman - V

      Rs.append(R)
      Vs.append(V.cpu().numpy())
      As.append(A.cpu().numpy())

      softmax = self.actor.forward(i)
      assert torch.eq(self.softmaxes[i], softmax.cpu()).all()
      entropy = -torch.sum(softmax * torch.log(softmax))
      actor_loss = -torch.log(softmax[0, self.sampled_action[i][0]]) * A
      actor_losses.append(actor_loss.detach().cpu().numpy())
      entropy_loss = FLAGS.entropy_weight / (entropy + 1e-6)
      entropy_losses.append(entropy_loss.detach().cpu().numpy())
      (actor_loss + entropy_loss).backward(retain_graph=i != 0)

      if (i + 1) % FLAGS.action_summary_frames == 0:
        ax1_height_ratio = 3
        fig, (ax1, ax2) = plt.subplots(2, gridspec_kw={
            'height_ratios' : [ax1_height_ratio, 1],
        })
        ax1.scatter(self.goal_x, self.goal_y, facecolors='none', edgecolors='r')
        last_frame = self.frame_buffer[i]
        trajectory_i = self.trajectory[max(0, i - FLAGS.context_frames):i+1]
        utils.plot_trajectory(last_frame, trajectory_i, ax=ax1)
        ax1.axis('off')

        num_topk = 5
        softmax_np = softmax[0].detach().cpu().numpy()
        topk_idxs = np.argsort(softmax_np)[::-1][:num_topk]
        labels = [','.join(utils.class2button(idx)) for idx in topk_idxs]
        ax2.bar(np.arange(num_topk), softmax_np[topk_idxs], width=0.3)
        ax2.set_xticks(np.arange(num_topk))
        ax2.set_xticklabels(labels)
        ax2.set_ylim(0.0, 1.0)
        asp = np.diff(ax2.get_xlim())[0] / np.diff(ax2.get_ylim())[0]
        asp /= np.abs(np.diff(ax1.get_xlim())[0] / np.diff(ax1.get_ylim())[0])
        ax2.set_aspect(asp / ax1_height_ratio)
        ax2.set_title('Sampled: %s (%0.2f%%)' % (
            ','.join(utils.class2button(self.sampled_action[i][0])),
            softmax[0, self.sampled_action[i][0]] * 100.0))
        GLOBAL.summary_writer.add_figure('action/frame_%03d' % i, fig, self.episode_number)

    GLOBAL.summary_writer.add_scalar('actor_grad_norm', utils.grad_norm(self.actor), self.episode_number)
    GLOBAL.summary_writer.add_scalar('critic_grad_norm', utils.grad_norm(self.critic), self.episode_number)
    assert not (FLAGS.clip_grad_value and FLAGS.clip_grad_norm)
    if FLAGS.clip_grad_value:
      nn.utils.clip_grad_value_(self.actor.parameters(), FLAGS.clip_grad_value)
      nn.utils.clip_grad_value_(self.critic.parameters(), FLAGS.clip_grad_value)
    elif FLAGS.clip_grad_norm:
      nn.utils.clip_grad_norm_(self.actor.parameters(), FLAGS.clip_grad_norm)
      nn.utils.clip_grad_norm_(self.critic.parameters(), FLAGS.clip_grad_norm)
    if self.episode_number >= FLAGS.actor_start_delay:
      self.optimizer_actor.step()
    self.optimizer_critic.step()

    fig = plt.figure()
    plt.plot(self.rewards)
    GLOBAL.summary_writer.add_figure('out/reward', fig, self.episode_number)
    fig = plt.figure()
    plt.plot(list(reversed(Rs[1:])))
    GLOBAL.summary_writer.add_figure('out/reward_cumul', fig, self.episode_number)
    fig = plt.figure()
    plt.plot(list(reversed(Vs[1:])))
    GLOBAL.summary_writer.add_figure('out/value', fig, self.episode_number)
    fig = plt.figure()
    plt.plot(list(reversed(As[1:])))
    GLOBAL.summary_writer.add_figure('out/advantage', fig, self.episode_number)
    fig = plt.figure()
    plt.plot(list(reversed(actor_losses)))
    GLOBAL.summary_writer.add_figure('loss/actor', fig, self.episode_number)
    fig = plt.figure()
    plt.plot(list(reversed(entropy_losses)))
    GLOBAL.summary_writer.add_figure('loss/entropy', fig, self.episode_number)

    # Start next episode.
    self.processed_frames = 0
    self.episode_number += 1
    if self.episode_number % FLAGS.save_every == 0:
      torch.save(self.actor.state_dict(), os.path.join(FLAGS.logdir,
          'train/celeste_model_actor_%d.pt' % self.episode_number))
      torch.save(self.critic.state_dict(), os.path.join(FLAGS.logdir,
          'train/celeste_model_critic_%d.pt' % self.episode_number))
      torch.save(self.actor.state_dict(), os.path.join(FLAGS.logdir,
          'train/celeste_model_actor_latest.pt'))
      torch.save(self.critic.state_dict(), os.path.join(FLAGS.logdir,
          'train/celeste_model_critic_latest.pt'))
    if self.episode_number >= FLAGS.max_episodes:
      return None
    return self.process_frame(frame=None)

  def process_frame(self, frame):
    """Returns a list of button inputs for the next N frames."""
    if FLAGS.interactive:
      button_input = []
      while not button_input:
        button_input = input('Buttons please! (comma separated)').split(',')
        if button_input == ['save']:
          self.savestate(0)
        elif button_input == ['load']:
          self.loadstate(0)
        elif button_input == ['start_episode']:
          FLAGS.interactive = False
          break
        else:
          button_inputs = [button_input]

    if not FLAGS.interactive:
      if self.processed_frames == 0:
        logging.info('Starting episode %d', self.episode_number)
        self.actor.reset()
        self.critic.reset()
        self.frame_buffer = []
        self.rewards = []
        self.softmaxes = []
        self.sampled_action = []
        self.trajectory = []
        self._generate_goal_state()
        if FLAGS.use_cuda:
          torch.cuda.empty_cache()

      if frame is None:
        if np.random.uniform() < FLAGS.random_loadstate_prob:
          custom_savestates = set(self.saved_states.keys())
          if len(custom_savestates) > 1:
            custom_savestates.remove(0)
          self.loadstate(int(np.random.choice(list(custom_savestates))))
        else:
          self.loadstate(0)
      else:
        self._set_inputs_from_frame(frame)
        if not self.saved_states:
          assert self.det.death_clock == 0
          # First (default) savestate.
          self.savestate(0)
        elif (FLAGS.num_custom_savestates
              and self.det.death_clock == 0
              and np.random.uniform() < FLAGS.random_savestate_prob):
          self.savestate(1 + np.random.randint(FLAGS.num_custom_savestates))

      if self.processed_frames > 0:
        reward, should_end_episode = self._reward_for_current_state()
        self.rewards.append(reward)
        if should_end_episode:
          return self._finish_episode()

      with torch.no_grad():
        softmax = self.actor.forward(self.processed_frames).detach().cpu()
      self.softmaxes.append(softmax)
      idxs, button_inputs = sample_action(softmax)
      self.sampled_action.append(idxs)
      # Predicted button_inputs include a batch dimension.
      button_inputs = button_inputs[0]
      # Returned button_inputs should be for next N frames, but for now N==1.
      button_inputs = [button_inputs] * FLAGS.hold_buttons_for

    self.processed_frames += 1
    return button_inputs


def Speedrun(env):
  moviefile = None
  if FLAGS.movie_file:
    moviefile = pylibtas.MovieFile()
    if moviefile.loadInputs(FLAGS.movie_file) != 0:
      raise ValueError('Could not load movie %s' % FLAGS.movie_file)

  processor = FrameProcessor(env)
  action_queue = queue.Queue()
  while True:
    frame = env.start_frame()

    ai = pylibtas.AllInputs()
    ai.emptyInputs()

    if moviefile and env.frame_counter < moviefile.nbFrames():
      moviefile.getInputs(ai, env.frame_counter)
    else:
      if action_queue.empty():
        for frame_actions in processor.process_frame(frame):
          action_queue.put(frame_actions)
      frame_actions = action_queue.get()
      assert isinstance(frame_actions, (list, tuple))
      for button in frame_actions:
        if button not in button_dict:
          logging.warning('Unknown button %s!' % button)
          continue
        si = pylibtas.SingleInput()
        si.type = button_dict[button]
        ai.setInput(si, 1)
    env.end_frame(ai)


def main(argv):
  del argv  # unused

  logging.info('\n%s', pprint.pformat(FLAGS.flag_values_dict()))

  os.makedirs(FLAGS.logdir, exist_ok=True)
  train_dir = os.path.join(FLAGS.logdir, 'train')
  os.makedirs(train_dir, exist_ok=True)
  GLOBAL.summary_writer = SummaryWriter(train_dir)

  tensorboard = subprocess.Popen(['tensorboard', '--logdir', os.path.abspath(FLAGS.logdir)],
                                 stderr=subprocess.DEVNULL)

  env = environment.Environment()
  try:
    if FLAGS.profile:
      FLAGS.max_episodes = 1
      cProfile.run('Speedrun(env)')
    else:
      Speedrun(env)
  finally:
    GLOBAL.summary_writer.close()
    if tensorboard:
      tensorboard.terminate()
    if env.game_pid != -1:
      logging.info('killing game %d' % env.game_pid)
      os.kill(env.game_pid, signal.SIGKILL)


if __name__ == '__main__':
  app.run(main)
