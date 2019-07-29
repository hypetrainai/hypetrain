from absl import app
from absl import flags
from absl import logging
logging.set_verbosity(logging.INFO)
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
flags.DEFINE_string('logdir', 'trained_models/bellsandwhistles', 'logdir')

flags.DEFINE_integer('save_every', 100, 'every X number of steps save a model')

flags.DEFINE_string('movie_file', 'movie.ltm', 'if not empty string, load libTAS input movie file')
flags.DEFINE_string('save_file', 'level1_screen4', 'if not empty string, use save file.')
flags.DEFINE_integer('goal_y', 107, 'goal pixel coordinate in y')
flags.DEFINE_integer('goal_x', 611, 'goal pixel coordinate in x')


flags.DEFINE_boolean('interactive', False, 'interactive mode (enter buttons on command line)')

flags.DEFINE_integer('image_height', 540, 'image height')
flags.DEFINE_integer('image_width', 960, 'image width')
flags.DEFINE_integer('image_channels', 5, 'image channels')

flags.DEFINE_integer('num_actions', 72, 'number of actions')

flags.DEFINE_float('lr', 0.0002, 'learning rate')
flags.DEFINE_float('entropy_weight', 0.05, 'weight for entropy loss')
flags.DEFINE_float('reward_decay_multiplier', 0.95, 'reward function decay multiplier')
flags.DEFINE_integer('episode_length', 200, 'episode length')
flags.DEFINE_integer('context_frames', 30, 'number of frames passed to the network')
flags.DEFINE_integer('bellman_lookahead_frames', 18, 'number of frames to consider for bellman rollout')

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
  assert pylibtas.receiveMessage() == pylibtas.MSGB_SAVING_SUCCEEDED


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

  assert msg == pylibtas.MSGB_FRAMECOUNT_TIME
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
    'x': pylibtas.SingleInput.IT_CONTROLLER1_BUTTON_X,
    'y': pylibtas.SingleInput.IT_CONTROLLER1_BUTTON_Y,
    'u': pylibtas.SingleInput.IT_CONTROLLER1_BUTTON_DPAD_UP,
    'd': pylibtas.SingleInput.IT_CONTROLLER1_BUTTON_DPAD_DOWN,
    'l': pylibtas.SingleInput.IT_CONTROLLER1_BUTTON_DPAD_LEFT,
    'r': pylibtas.SingleInput.IT_CONTROLLER1_BUTTON_DPAD_RIGHT,
    'rt': pylibtas.SingleInput.IT_CONTROLLER1_BUTTON_RIGHTSHOULDER,
    'lt': pylibtas.SingleInput.IT_CONTROLLER1_BUTTON_LEFTSHOULDER
}


def sample_action(softmax):
    softmax = softmax.detach().cpu()
    sample = torch.distributions.categorical.Categorical(probs=softmax).sample()
    sample_mapped = [utils.class2button(int(sample[i].numpy())) for i in range(len(sample))]
    return sample, sample_mapped

def generate_gaussian_heat_map(image_shape, y, x, sigma=10, amplitude=1.0):
    H, W = image_shape
    y_range = np.arange(0, H)
    x_range = np.arange(0, W)
    x_grid, y_grid = np.meshgrid(x_range, y_range)

    result = np.zeros([1, 1, H, W])
    result[0, 0] = amplitude * np.exp((-1.0 * (y_grid - y)**2 + -1.0 * (x_grid - x)**2)/(2 * sigma**2))
    return result

class FrameProcessor(object):

  def __init__(self):
    self.det = celeste_detector.CelesteDetector()

    self.goal_y = FLAGS.goal_y
    self.goal_x = FLAGS.goal_x
    self.episode_number = 0
    gigit  = -1
    self.episode_start = -1
    self.start_frame = None
    self.frame_buffer = None
    self.trajectory = []
    self.rewards = []
    self.sampled_action = []

    fake_inputs = torch.zeros(1, FLAGS.image_channels * FLAGS.context_frames,
                              FLAGS.image_height, FLAGS.image_width)
    self.actor = Network()
    # GLOBAL.summary_writer.add_graph(self.actor, fake_inputs)
    self.actor = nn.DataParallel(self.actor.cuda())
    self.critic = Network(out_dim=1, use_softmax=False)
    # GLOBAL.summary_writer.add_graph(self.critic, fake_inputs)
    self.critic = nn.DataParallel(self.critic.cuda())
    self.optimizer_actor = optim.Adam(list(self.actor.parameters()), lr=FLAGS.lr)
    self.optimizer_critic = optim.Adam(list(self.critic.parameters()), lr=FLAGS.lr)

    self.start_frame_saving = False
    self.saved_frames = 0

    if FLAGS.pretrained_model_path:
        logging.info('Loading pretrained model from %s' % FLAGS.pretrained_model_path)
        self.actor.load_state_dict(torch.load(
            FLAGS.pretrained_model_path + '/celeste_model_actor_%s.pt' % FLAGS.pretrained_suffix))
        self.critic.load_state_dict(torch.load(
            FLAGS.pretrained_model_path + '/celeste_model_critic_%s.pt' % FLAGS.pretrained_suffix))
        logging.info('Done!')

  def _generate_goal_state(self, custom_goal = None):
    if custom_goal:
        self.goal_y = custom_goal[0]
        self.goal_x = custom_goal[1]
    else:
        self.goal_y = FLAGS.goal_y
        self.goal_x = FLAGS.goal_x

  def _reward_function_for_current_state(self):
    """Returns (rewards, should_end_episode) given state."""
    y, x = self.trajectory[-1]
    if y is None:
      # Assume death
      return 0, True  # TODO: death penalty
    dist_to_goal = np.sqrt((y - self.goal_y)**2 + (x - self.goal_x)**2)
    reward = -1.0 * dist_to_goal
    if dist_to_goal < 10:
      return reward + 100, True
    if frame_counter - self.episode_start >= FLAGS.episode_length:
      return reward, True
    return reward, False

  def _finish_episode(self):
    assert self.episode_start >= 0
    num_frames = frame_counter - self.episode_start
    assert len(self.rewards) == num_frames, (num_frames, len(self.rewards))
    assert len(self.sampled_action) == num_frames, (num_frames, len(self.sampled_action))
    assert len(self.trajectory) == num_frames + 1, (num_frames + 1, len(self.trajectory))
    assert self.frame_buffer.shape[1] == num_frames + FLAGS.context_frames, (num_frames, self.frame_buffer.shape[1])

    GLOBAL.summary_writer.add_scalar('episode_length', num_frames, self.episode_number)
    GLOBAL.summary_writer.add_scalar('final_reward', self.rewards[-1], self.episode_number)
    GLOBAL.summary_writer.add_scalar('best_reward', max(self.rewards), self.episode_number)
    # GLOBAL.summary_writer.add_video('input_frames', self.frame_buffer[:, :, :3], self.episode_number, fps=60)

    average_frame = np.mean(self.frame_buffer[0, -(num_frames+1):, :3].detach().cpu().numpy(), axis=0)
    plt.figure()
    plt.scatter(self.goal_x, self.goal_y, facecolors='none', edgecolors='r')
    utils.plotTrajectory(average_frame, self.trajectory)
    GLOBAL.summary_writer.add_figure('trajectory', plt.gcf(), self.episode_number)

    R = 0
    Vs = []
    Rs = []
    As = []
    actor_losses = []
    self.optimizer_actor.zero_grad()
    for i in reversed(range(num_frames)):
      frames = self.frame_buffer[:, i:i+FLAGS.context_frames]
      frames = torch.reshape(frames, [1, -1, FLAGS.image_height, FLAGS.image_width])
      V = self.critic.forward(frames).view([])
      R = FLAGS.reward_decay_multiplier * R + self.rewards[i]

      if FLAGS.bellman_lookahead_frames == 0 or i == num_frames - 1:
        A = R - V
      else:
        blf = min(FLAGS.bellman_lookahead_frames, num_frames - 1 - i)
        assert blf > 0
        V_bellman = R - (FLAGS.reward_decay_multiplier**blf) * Rs[-blf] + Vs[-blf]
        A = V_bellman - V

      Vs.append(V.detach().cpu().numpy())
      Rs.append(R)
      As.append(A.detach().cpu().numpy())

      self.optimizer_critic.zero_grad()
      (A**2).backward(retain_graph=True)
      self.optimizer_critic.step()

      softmax = self.actor.forward(frames)
      entropy = torch.distributions.categorical.Categorical(probs=softmax).entropy()
      actor_loss = (-1.0 * torch.log(softmax[0, self.sampled_action[i]]) * A
                    - FLAGS.entropy_weight * entropy)
      actor_losses.append(actor_loss.detach().cpu().numpy())
      actor_loss.backward()
    self.optimizer_actor.step()

    plt.figure()
    plt.plot(list(reversed(Vs)))
    GLOBAL.summary_writer.add_figure("loss/value", plt.gcf(), self.episode_number)
    plt.figure()
    plt.plot(list(reversed(Rs)))
    GLOBAL.summary_writer.add_figure("loss/reward", plt.gcf(), self.episode_number)
    plt.figure()
    plt.plot(list(reversed(As)))
    GLOBAL.summary_writer.add_figure("loss/advantage", plt.gcf(), self.episode_number)
    plt.figure()
    plt.plot(list(reversed(actor_losses)))
    GLOBAL.summary_writer.add_figure("loss/actor_loss", plt.gcf(), self.episode_number)

    # Start next episode.
    loadstate()
    self.episode_start = -1
    self.episode_number += 1
    logging.info('Starting episode %d', self.episode_number)
    if self.episode_number % FLAGS.save_every == 0:
        torch.save(self.actor.state_dict(), os.path.join(FLAGS.logdir,
            'train/celeste_model_actor_%d.pt' % self.episode_number))
        torch.save(self.critic.state_dict(), os.path.join(FLAGS.logdir,
            'train/celeste_model_critic_%d.pt' % self.episode_number))
        torch.save(self.actor.state_dict(), os.path.join(FLAGS.logdir,
            'train/celeste_model_actor_latest.pt'))
        torch.save(self.critic.state_dict(), os.path.join(FLAGS.logdir,
            'train/celeste_model_critic_latest.pt'))
    return self.process_frame(self.start_frame)

  def process_frame(self, frame):
    if FLAGS.interactive:
      button_input = input('Buttons please! (comma separated)').split(',')
      button_input = []
      if button_input:
        if button_input == ['save']:
          savestate()
        elif button_input == ['load']:
          loadstate()
        elif button_input == ['start_episode']:
          FLAGS.interactive = False
        if button_input[-1] == 'sf':
          self.start_frame_saving = True
          button_input = button_input[:-1]
      if self.start_frame_saving:
        imageio.imwrite('frame_%04d.png' % self.saved_frames, frame)
        self.saved_frames += 1

    if not FLAGS.interactive:
      cuda_frame = torch.tensor(frame).float().permute(2, 0, 1).unsqueeze(0).cuda() / 255.0

      if self.episode_start < 0:
        if self.start_frame is None:
          savestate()
        self.start_frame = frame
        self.rewards = []
        self.sampled_action = []
        self._generate_goal_state()
        prior_coord = None
      else:
        prior_coord = self.trajectory[-1]
      y, x, state = self.det.detect(frame, prior_coord=prior_coord)
    
      # generate the full frame input by concatenating gaussian heat maps. 
      if y is None:
        gaussian_current_position = torch.zeros(frame[:, :, 0].shape).cuda()
      else:
        gaussian_current_position = torch.tensor(generate_gaussian_heat_map(frame[:, :, 0].shape, y, x)).float().cuda()
      gaussian_goal_position = torch.tensor(generate_gaussian_heat_map(frame[:, :, 0].shape, self.goal_y, self.goal_x)).float().cuda()
      cuda_frame = torch.cat([cuda_frame, gaussian_current_position, gaussian_goal_position], 1)

      if self.episode_start < 0:
        assert state != -1
        self.frame_buffer = torch.stack([cuda_frame] * FLAGS.context_frames, 1)
        self.trajectory = [(y, x)]
        self.episode_start = frame_counter
      else:
        self.frame_buffer = torch.cat([self.frame_buffer, cuda_frame.unsqueeze(1)], 1)
        self.trajectory.append((y, x))
        reward, should_end_episode = self._reward_function_for_current_state()
        self.rewards.append(reward)
        if should_end_episode:
          return self._finish_episode()

      frames = self.frame_buffer[:, -FLAGS.context_frames:]
      frames = torch.reshape(frames, [1, -1, FLAGS.image_height, FLAGS.image_width])
      softmax = self.actor.forward(frames)
      idx, button_input = sample_action(softmax)
      self.sampled_action.append(idx)

    return button_input


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
    assert msg == pylibtas.MSGB_FRAME_DATA, msg
    _, actual_window_width = pylibtas.receiveInt()
    _, actual_window_height = pylibtas.receiveInt()
    assert actual_window_width == FLAGS.image_width and actual_window_height == FLAGS.image_height
    _, size = pylibtas.receiveInt()
    received, frame = pylibtas.receiveArray(size)
    assert received == size, (size, received)

    frame = np.reshape(frame, [FLAGS.image_height, FLAGS.image_width, 4])[:, :, :3]

    if frame_counter == 0:
      pylibtas.sendMessage(pylibtas.MSGN_SAVESTATE_PATH)
      pylibtas.sendString('/tmp/celeste/savestate')

    ai = pylibtas.AllInputs()
    ai.emptyInputs()
    if moviefile and frame_counter < moviefile.nbFrames():
      moviefile.getInputs(ai, frame_counter)
    else:
      if action_queue.empty():
        for button_inputs in processor.process_frame(frame):
          action_queue.put(button_inputs)
      button_input = action_queue.get()
      for button in button_input:
        if button not in button_dict:
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
    Speedrun()
  finally:
    GLOBAL.summary_writer.close()
    if tensorboard:
      tensorboard.terminate()
    if game_pid != -1:
      logging.info('killing game %d' % game_pid)
      os.kill(game_pid, signal.SIGKILL)


if __name__ == "__main__":
  app.run(main)
