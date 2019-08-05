from absl import app
from absl import flags
from absl import logging
logging.set_verbosity(logging.INFO)
import cProfile
import imageio
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
from torch import optim

from GLOBALS import GLOBAL
import celeste_detector
from model import ResNetIm2Value as Network
import utils


flags.DEFINE_string('pretrained_model_path', '', 'pretrained model path')
flags.DEFINE_string('pretrained_suffix', 'latest', 'if latest, will load most recent save in dir')
flags.DEFINE_string('logdir', 'trained_models/randomgoaltest3', 'logdir')
flags.DEFINE_boolean('use_cuda', True, 'Use cuda')
flags.DEFINE_boolean('profile', False, 'Profile code')

flags.DEFINE_integer('max_episodes', 100000, 'stop after this many episodes')
flags.DEFINE_integer('save_every', 100, 'every X number of steps save a model')

flags.DEFINE_string('movie_file', 'movie.ltm', 'if not empty string, load libTAS input movie file')
flags.DEFINE_string('save_file', 'level1_screen4', 'if not empty string, use save file.')
flags.DEFINE_integer('goal_y', 107, 'goal pixel coordinate in y')
flags.DEFINE_integer('goal_x', 611, 'goal pixel coordinate in x')

flags.DEFINE_boolean('interactive', False, 'interactive mode (enter buttons on command line)')

flags.DEFINE_integer('image_height', 540, 'image height')
flags.DEFINE_integer('image_width', 960, 'image width')

flags.DEFINE_float('lr', 0.001, 'learning rate')
flags.DEFINE_float('entropy_weight', 0.5, 'weight for entropy loss')
flags.DEFINE_float('reward_scale', 1.0/100.0, 'multiplicative scale for the reward function')
flags.DEFINE_float('reward_decay_multiplier', 0.95, 'reward time decay multiplier')
flags.DEFINE_integer('episode_length', 200, 'episode length')
flags.DEFINE_integer('context_frames', 30, 'number of frames passed to the network')
flags.DEFINE_integer('bellman_lookahead_frames', 12, 'number of frames to consider for bellman rollout')
flags.DEFINE_float('clip_grad_norm', 1000.0, 'value to clip gradient norm to.')
flags.DEFINE_float('clip_grad_value', 0.0, 'value to clip gradients to.')

flags.DEFINE_float('random_goal_probability', 0.4, 'probability that we choose a random goal')
flags.DEFINE_integer('action_summary_frames', 50, 'number of frames between action summaries')

FLAGS = flags.FLAGS


SIZE_INT = 4
SIZE_FLOAT = 4
SIZE_UNSIGNED_LONG = 8
SIZE_TIMESPEC = 16
SIZE_GAMEINFO_STRUCT = 36


shared_config = None
game_pid = -1
frame_counter = 0


def savestate(index=1):
  pylibtas.sendMessage(pylibtas.MSGN_SAVESTATE_INDEX)
  pylibtas.sendInt(index)
  pylibtas.sendMessage(pylibtas.MSGN_SAVESTATE)
  utils.assert_equal(pylibtas.receiveMessage(), pylibtas.MSGB_SAVING_SUCCEEDED)


# TODO: move these functions all into a class so shared_config can be a class member.
def loadstate(index=1):
  global frame_counter
  pylibtas.sendMessage(pylibtas.MSGN_SAVESTATE_INDEX)
  pylibtas.sendInt(index)
  pylibtas.sendMessage(pylibtas.MSGN_LOADSTATE)

  msg = pylibtas.receiveMessage()
  if msg == pylibtas.MSGB_LOADING_SUCCEEDED:
    pylibtas.sendMessage(pylibtas.MSGN_CONFIG)
    pylibtas.sendSharedConfig(shared_config)
    msg = pylibtas.receiveMessage()

  utils.assert_equal(msg, pylibtas.MSGB_FRAMECOUNT_TIME)
  _, frame_counter = pylibtas.receiveULong()
  pylibtas.ignoreData(SIZE_TIMESPEC)
  pylibtas.sendMessage(pylibtas.MSGN_EXPOSE)


def start_next_frame():
  msg = pylibtas.receiveMessage()
  while msg != pylibtas.MSGB_START_FRAMEBOUNDARY:
    if msg == pylibtas.MSGB_WINDOW_ID:
      pylibtas.ignoreData(SIZE_INT)
    elif msg == pylibtas.MSGB_ALERT_MSG:
      logging.warning(pylibtas.receiveString())
    elif msg == pylibtas.MSGB_ENCODE_FAILED:
      raise RuntimeError('MSGB_ENCODE_FAILED')
    elif msg == pylibtas.MSGB_FRAMECOUNT_TIME:
      pylibtas.ignoreData(SIZE_UNSIGNED_LONG)
      pylibtas.ignoreData(SIZE_TIMESPEC)
    elif msg == pylibtas.MSGB_GAMEINFO:
      pylibtas.ignoreData(SIZE_GAMEINFO_STRUCT)
    elif msg == pylibtas.MSGB_FPS:
      pylibtas.ignoreData(SIZE_FLOAT)
      pylibtas.ignoreData(SIZE_FLOAT)
    elif msg == pylibtas.MSGB_ENCODING_SEGMENT:
      pylibtas.ignoreData(SIZE_INT)
    elif msg == pylibtas.MSGB_QUIT:
      raise RuntimeError('User Quit.')
    elif msg == -1:
      raise RuntimeError('The connection to the game was lost.')
    else:
      raise RuntimeError('Received unexpected message %s(%d)' % (pylibtas.message_name(msg), msg))
    msg = pylibtas.receiveMessage()
  pylibtas.sendMessage(pylibtas.MSGN_START_FRAMEBOUNDARY)


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
    return torch.tensor(result).unsqueeze(0).float()


class FrameProcessor(object):

  def __init__(self):
    self.det = celeste_detector.CelesteDetector()

    self.episode_number = 0
    self.episode_start = -1
    self.start_frame = None

    in_dim = 4 * FLAGS.context_frames + 1 + len(button_dict)
    self.actor = Network(in_dim, out_dim=len(utils.class2button.dict))
    self.critic = Network(in_dim, out_dim=1, use_softmax=False)
    if FLAGS.use_cuda:
      self.actor = self.actor.cuda()
      self.critic = self.critic.cuda()
    self.actor = nn.DataParallel(self.actor)
    self.critic = nn.DataParallel(self.critic)
    self.optimizer_actor = optim.Adam(list(self.actor.parameters()), lr=FLAGS.lr)
    self.optimizer_critic = optim.Adam(list(self.critic.parameters()), lr=FLAGS.lr)

    if FLAGS.pretrained_model_path:
        logging.info('Loading pretrained model from %s' % FLAGS.pretrained_model_path)
        self.actor.load_state_dict(torch.load(
            FLAGS.pretrained_model_path + '/celeste_model_actor_%s.pt' % FLAGS.pretrained_suffix))
        self.critic.load_state_dict(torch.load(
            FLAGS.pretrained_model_path + '/celeste_model_critic_%s.pt' % FLAGS.pretrained_suffix))
        logging.info('Done!')

  def _generate_goal_state(self):
    if np.random.uniform() < FLAGS.random_goal_probability:
        self.goal_y = np.random.randint(50, FLAGS.image_height - 50)
        self.goal_x = np.random.randint(50, FLAGS.image_width - 50)
    else:
        self.goal_y = FLAGS.goal_y
        self.goal_x = FLAGS.goal_x

  def _reward_function_for_current_state(self):
    """Returns (rewards, should_end_episode) given state."""
    reward = 0
    should_end_episode = False
    y, x = self.trajectory[-1]
    if y is None:
      # Assume death
      reward -= 25
      should_end_episode = True
      y, x = self.trajectory[-2]
    dist_to_goal = np.sqrt((y - self.goal_y)**2 + (x - self.goal_x)**2)
    reward -= dist_to_goal
    if dist_to_goal < 10 and not should_end_episode:
      reward += 100
      should_end_episode = True
    if frame_counter - self.episode_start >= FLAGS.episode_length:
      should_end_episode = True
    return reward * FLAGS.reward_scale, should_end_episode

  def _get_network_inputs(self, i):
    input_frames = self.frame_buffer[i:i+FLAGS.context_frames]
    # [time, channels, height, width] -> [time * channels, height, width]
    input_frames = torch.reshape(input_frames, [-1, FLAGS.image_height, FLAGS.image_width])
    input_frames = torch.cat([input_frames, self.extra_channels[i]], 0)
    input_frames = input_frames.unsqueeze(0)
    return input_frames

  def _start_new_episode(self, frame):
    logging.info('Starting episode %d', self.episode_number)
    if self.start_frame is None:
      savestate()
    self.episode_start = frame_counter
    self.start_frame = frame
    self.rewards = []
    self.softmaxes = []
    self.sampled_action = []
    self.trajectory = []
    self.frame_buffer = None
    self.extra_channels = []
    self._generate_goal_state()
    torch.cuda.empty_cache()

  def _finish_episode(self):
    assert self.episode_start >= 0
    num_frames = frame_counter - self.episode_start
    utils.assert_equal(len(self.trajectory), num_frames + 1)
    utils.assert_equal(self.frame_buffer.shape[0], num_frames + FLAGS.context_frames)
    utils.assert_equal(len(self.extra_channels), num_frames)
    utils.assert_equal(len(self.rewards), num_frames)
    utils.assert_equal(len(self.softmaxes), num_frames)
    utils.assert_equal(len(self.sampled_action), num_frames)

    GLOBAL.summary_writer.add_scalar('episode_length', num_frames, self.episode_number)
    GLOBAL.summary_writer.add_scalar('final_reward', self.rewards[-1], self.episode_number)
    GLOBAL.summary_writer.add_scalar('best_reward', max(self.rewards), self.episode_number)
    # GLOBAL.summary_writer.add_video('input_frames', self.frame_buffer[:, :3], self.episode_number, fps=60)

    plt.figure()
    plt.scatter(self.goal_x, self.goal_y, facecolors='none', edgecolors='r')
    utils.plot_trajectory(self.frame_buffer[-1, :3].detach().cpu().numpy(), self.trajectory)
    GLOBAL.summary_writer.add_figure('trajectory', plt.gcf(), self.episode_number)

    R = 0
    Rs = []
    Vs = []
    As = []
    actor_losses = []
    entropy_losses = []
    self.optimizer_actor.zero_grad()
    self.optimizer_critic.zero_grad()
    for i in reversed(range(num_frames)):
      R = FLAGS.reward_decay_multiplier * R + self.rewards[i]
      V = self.critic.forward(self._get_network_inputs(i)).view([])

      if FLAGS.bellman_lookahead_frames == 0 or i == num_frames - 1:
        A = R - V
      else:
        blf = min(FLAGS.bellman_lookahead_frames, num_frames - 1 - i)
        assert blf > 0
        V_bellman = (R - (FLAGS.reward_decay_multiplier**blf) * Rs[-blf]
                     + (FLAGS.reward_decay_multiplier**blf) * Vs[-blf])
        A = V_bellman - V

      (A**2).backward()
      V = V.detach()
      A = A.detach()

      Rs.append(R)
      Vs.append(V.cpu().numpy())
      As.append(A.cpu().numpy())

      softmax = self.actor.forward(self._get_network_inputs(i))
      assert torch.eq(self.softmaxes[i], softmax.cpu()).all()
      entropy = torch.distributions.categorical.Categorical(probs=softmax).entropy()
      actor_loss = -torch.log(softmax[0, self.sampled_action[i][0]]) * A
      actor_losses.append(actor_loss.detach().cpu().numpy())
      entropy_loss = FLAGS.entropy_weight * -torch.log(entropy)
      entropy_losses.append(entropy_loss.detach().cpu().numpy())
      (actor_loss + entropy_loss).backward()

      if (i + 1) % FLAGS.action_summary_frames == 0:
        ax1_height_ratio = 3
        fig, (ax1, ax2) = plt.subplots(2, gridspec_kw={
            'height_ratios' : [ax1_height_ratio, 1],
        })
        ax1.scatter(self.goal_x, self.goal_y, facecolors='none', edgecolors='r')
        last_frame = self.frame_buffer[i + FLAGS.context_frames - 1, :3].detach().cpu().numpy()
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
    self.optimizer_actor.step()
    self.optimizer_critic.step()

    plt.figure()
    plt.plot(self.rewards)
    GLOBAL.summary_writer.add_figure('out/reward', plt.gcf(), self.episode_number)
    plt.figure()
    plt.plot(list(reversed(Rs)))
    GLOBAL.summary_writer.add_figure('out/reward_cumul', plt.gcf(), self.episode_number)
    plt.figure()
    plt.plot(list(reversed(Vs)))
    GLOBAL.summary_writer.add_figure('out/value', plt.gcf(), self.episode_number)
    plt.figure()
    plt.plot(list(reversed(As)))
    GLOBAL.summary_writer.add_figure('out/advantage', plt.gcf(), self.episode_number)
    plt.figure()
    plt.plot(list(reversed(actor_losses)))
    GLOBAL.summary_writer.add_figure('loss/actor', plt.gcf(), self.episode_number)
    plt.figure()
    plt.plot(list(reversed(entropy_losses)))
    GLOBAL.summary_writer.add_figure('loss/entropy', plt.gcf(), self.episode_number)

    # Start next episode.
    loadstate()
    self.episode_start = -1
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
    return self.process_frame(self.start_frame)

  def process_frame(self, frame):
    """Returns a list of button inputs for the next N frames."""
    if FLAGS.interactive:
      button_input = []
      while not button_input:
        button_input = input('Buttons please! (comma separated)').split(',')
        if button_input == ['save']:
          savestate()
        elif button_input == ['load']:
          loadstate()
        elif button_input == ['start_episode']:
          FLAGS.interactive = False
          break
        else:
          button_inputs = [button_input]

    if not FLAGS.interactive:
      new_episode = False
      if self.episode_start < 0:
        new_episode = True
        self._start_new_episode(frame)

      y, x, state = self.det.detect(frame, prior_coord=None if new_episode else self.trajectory[-1])
      self.trajectory.append((y, x))

      window_shape = [FLAGS.image_height, FLAGS.image_width]
      # generate the full frame input by concatenating gaussian heat maps.
      if y is None:
        gaussian_current_position = torch.zeros(window_shape).unsqueeze(0)
      else:
        gaussian_current_position = generate_gaussian_heat_map(window_shape, y, x)

      input_frame = torch.tensor(frame).float().permute(2, 0, 1) / 255.0
      input_frame = torch.cat([input_frame, gaussian_current_position], 0)
      if FLAGS.use_cuda:
        input_frame = input_frame.cuda()

      if new_episode:
        assert state != -1
        self.frame_buffer = torch.stack([input_frame] * FLAGS.context_frames, 0)
      else:
        self.frame_buffer = torch.cat([self.frame_buffer, input_frame.unsqueeze(0)], 0)
        reward, should_end_episode = self._reward_function_for_current_state()
        self.rewards.append(reward)
        if should_end_episode:
          return self._finish_episode()

      gaussian_goal_position = generate_gaussian_heat_map(window_shape, self.goal_y, self.goal_x)

      last_frame_buttons = torch.zeros([len(button_dict)] + window_shape)
      if self.sampled_action:
        pressed = utils.class2button(self.sampled_action[-1][0])
        for i, button in enumerate(button_dict.keys()):
          if button in pressed:
            last_frame_buttons[i] = 1.0

      extra_channels = torch.cat([gaussian_goal_position, last_frame_buttons], 0)
      if FLAGS.use_cuda:
        extra_channels = extra_channels.cuda()
      self.extra_channels.append(extra_channels)

      frame_id = frame_counter - self.episode_start
      utils.assert_equal(frame_id + FLAGS.context_frames, self.frame_buffer.shape[0])
      utils.assert_equal(frame_id, len(self.extra_channels) - 1)
      with torch.no_grad():
        softmax = self.actor.forward(self._get_network_inputs(frame_id)).detach().cpu()
      self.softmaxes.append(softmax)
      idxs, button_inputs = sample_action(softmax)
      # Predicted button_inputs include a batch dimension.
      # Returned button_inputs should be for next N frames, but for now N==1.
      button_inputs = [button_inputs[0]]
      self.sampled_action.append(idxs)

    return button_inputs


def Speedrun():
  global frame_counter
  global shared_config

  os.system('mkdir -p /tmp/celeste/movies')
  os.system('cp -f settings.celeste ~/.local/share/Celeste/Saves/')
  moviefile = None
  if FLAGS.movie_file is not None:
    moviefile = pylibtas.MovieFile()
    if moviefile.loadInputs(FLAGS.movie_file) != 0:
      raise ValueError('Could not load movie %s' % FLAGS.movie_file)
  if FLAGS.save_file is not None:
    savepath = 'savefiles/' + FLAGS.save_file
    os.system('cp -f %s ~/.local/share/Celeste/Saves/0.celeste' % savepath)

  pylibtas.removeSocket()
  pylibtas.launchGameThread(
      'CelesteLinux/Celeste.bin.x86_64',
      'libTAS/build64/libtas.so',
      '',  # gameargs
      0,  # startframe
      'lib64',
      os.path.dirname(os.path.abspath(__file__)),
      pylibtas.SharedConfig.LOGGING_TO_CONSOLE,
      True,  # opengl_soft
      '',  # llvm_perf
      False,  # attach_gdb
  )
  pylibtas.initSocketProgram()

  msg = pylibtas.receiveMessage()
  while msg != pylibtas.MSGB_END_INIT:
    if msg == pylibtas.MSGB_PID:
      global game_pid
      _, game_pid = pylibtas.receiveInt()
    elif hasattr(pylibtas, 'MSGB_GIT_COMMIT') and msg == pylibtas.MSGB_GIT_COMMIT:
      _ = pylibtas.receiveString()
    else:
      raise RuntimeError('Unexpected message %d in init!' % msg)
    msg = pylibtas.receiveMessage()

  pylibtas.sendMessage(pylibtas.MSGN_CONFIG)
  shared_config = pylibtas.SharedConfig()
  shared_config.nb_controllers = 1
  shared_config.audio_mute = True
  shared_config.incremental_savestates = False
  shared_config.savestates_in_ram = True
  shared_config.backtrack_savestate = False
  shared_config.prevent_savefiles = False
  shared_config.recycle_threads = False
  shared_config.write_savefiles_on_exit = False
  shared_config.main_gettimes_threshold = [-1, -1, -1, 100, -1, -1]
  shared_config.includeFlags = pylibtas.LCF_ERROR
  pylibtas.sendSharedConfig(shared_config)

  pylibtas.sendMessage(pylibtas.MSGN_ENCODING_SEGMENT)
  pylibtas.sendInt(0)

  pylibtas.sendMessage(pylibtas.MSGN_END_INIT)

  processor = FrameProcessor()
  action_queue = queue.Queue()
  while True:
    start_next_frame()

    msg = pylibtas.receiveMessage()
    utils.assert_equal(msg, pylibtas.MSGB_FRAME_DATA)
    _, actual_window_width = pylibtas.receiveInt()
    utils.assert_equal(actual_window_width, FLAGS.image_width)
    _, actual_window_height = pylibtas.receiveInt()
    utils.assert_equal(actual_window_height, FLAGS.image_height)
    _, size = pylibtas.receiveInt()
    received, frame = pylibtas.receiveArray(size)
    utils.assert_equal(received, size)

    frame = np.reshape(frame, [FLAGS.image_height, FLAGS.image_width, 4])[:, :, :3]

    if frame_counter == 0:
      pylibtas.sendMessage(pylibtas.MSGN_SAVESTATE_PATH)
      pylibtas.sendString('/tmp/celeste/savestate')

    ai = pylibtas.AllInputs()
    ai.emptyInputs()
    if moviefile and frame_counter < moviefile.nbFrames():
      moviefile.getInputs(ai, frame_counter)
    else:
      frames_actions = processor.process_frame(frame)
      if not frames_actions:
        return
      if action_queue.empty():
        for frame_actions in frames_actions:
          action_queue.put(frame_actions)
      frame_actions = action_queue.get()
      assert isinstance(frame_actions, (list, tuple))
      if list(frame_actions) != ['']:
        for button in frame_actions:
          if button not in button_dict:
            logging.warning('Unknown button %s!' % button)
            continue
          si = pylibtas.SingleInput()
          si.type = button_dict[button]
          ai.setInput(si, 1)

    pylibtas.sendMessage(pylibtas.MSGN_ALL_INPUTS)
    pylibtas.sendAllInputs(ai)
    pylibtas.sendMessage(pylibtas.MSGN_END_FRAMEBOUNDARY)
    frame_counter += 1


def main(argv):
  del argv  # unused

  logging.info('\n%s', pprint.pformat(FLAGS.flag_values_dict()))

  os.makedirs(FLAGS.logdir, exist_ok=True)
  train_dir = os.path.join(FLAGS.logdir, 'train')
  os.makedirs(train_dir, exist_ok=True)
  GLOBAL.summary_writer = SummaryWriter(train_dir)

  tensorboard = subprocess.Popen(['tensorboard', '--logdir', os.path.abspath(FLAGS.logdir)],
                                 stderr=subprocess.DEVNULL)
  try:
    if FLAGS.profile:
      FLAGS.max_episodes = 1
      cProfile.run('Speedrun()')
    else:
      Speedrun()
  finally:
    GLOBAL.summary_writer.close()
    if tensorboard:
      tensorboard.terminate()
    if game_pid != -1:
      logging.info('killing game %d' % game_pid)
      os.kill(game_pid, signal.SIGKILL)


if __name__ == '__main__':
  app.run(main)
