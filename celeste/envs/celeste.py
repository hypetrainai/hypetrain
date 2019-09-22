from absl import flags
from absl import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import signal
import torch

from GLOBALS import GLOBAL
import celeste_detector
import env
import pylibtas
import utils

FLAGS = flags.FLAGS

_SIZE_INT = 4
_SIZE_FLOAT = 4
_SIZE_UNSIGNED_LONG = 8
_SIZE_TIMESPEC = 16
_SIZE_GAMEINFO_STRUCT = 36

_BUTTON_DICT = {
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

_ACTION_BUTTONS = [
  [''],
  ['a'],
  ['b'],
  ['rt'],
  ['a','b'],
  ['a','rt'],
  ['rt', 'b'],
  ['a','b','rt']
]

_DPAD_BUTTONS = [
  [''],
  ['r'],
  ['l'],
  ['u'],
  ['d'],
  ['r','u'],
  ['u','l'],
  ['l','d'],
  ['d','r']
]

_GOAL_MAP = {
    'level1_screen0': (152, 786),
    'level1_screen4': (107, 611),
}


class Env(env.Env):

  def __init__(self):
    super(Env, self).__init__()

    self.frame_counter = 0
    self.det = celeste_detector.CelesteDetector()

    self.class2button = {}
    for action_id, action_list in enumerate(_ACTION_BUTTONS):
      for dpad_id, dpad_list in enumerate(_DPAD_BUTTONS):
        final_id = action_id * len(_DPAD_BUTTONS) + dpad_id
        buttons = [button for button in action_list + dpad_list if button]
        self.class2button[final_id] = buttons

    os.system('mkdir -p /tmp/celeste/movies')
    os.system('cp -f settings.celeste ~/.local/share/Celeste/Saves/')
    if FLAGS.save_file is not None:
      savepath = 'savefiles/' + FLAGS.save_file
      os.system('cp -f %s ~/.local/share/Celeste/Saves/0.celeste' % savepath)

    self.moviefile = None
    if FLAGS.movie_file:
      self.moviefile = pylibtas.MovieFile()
      if self.moviefile.loadInputs(FLAGS.movie_file) != 0:
        raise ValueError('Could not load movie %s' % FLAGS.movie_file)

    pylibtas.removeSocket()
    pylibtas.launchGameThread(
        '../CelesteLinux/Celeste.bin.x86_64',
        '../third_party/libTAS/build64/libtas.so',
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
        _, self.game_pid = pylibtas.receiveInt()
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
    self.shared_config = shared_config

    pylibtas.sendMessage(pylibtas.MSGN_ENCODING_SEGMENT)
    pylibtas.sendInt(0)

    pylibtas.sendMessage(pylibtas.MSGN_END_INIT)

  def quit(self):
    if self.game_pid != -1:
      logging.info('killing game %d' % self.game_pid)
      os.kill(self.game_pid, signal.SIGKILL)

  def can_savestate(self):
    return self.det.death_clock == 0

  def savestate(self, index):
    pylibtas.sendMessage(pylibtas.MSGN_SAVESTATE_INDEX)
    pylibtas.sendInt(index)
    pylibtas.sendMessage(pylibtas.MSGN_SAVESTATE)
    utils.assert_equal(pylibtas.receiveMessage(), pylibtas.MSGB_SAVING_SUCCEEDED)

    self.det.savestate(index)
    self.saved_states[index] = (self.trajectory[-1:], self.frame_buffer[-1:])

  def loadstate(self, index):
    pylibtas.sendMessage(pylibtas.MSGN_SAVESTATE_INDEX)
    pylibtas.sendInt(index)
    pylibtas.sendMessage(pylibtas.MSGN_LOADSTATE)

    msg = pylibtas.receiveMessage()
    if msg == pylibtas.MSGB_LOADING_SUCCEEDED:
      pylibtas.sendMessage(pylibtas.MSGN_CONFIG)
      pylibtas.sendSharedConfig(self.shared_config)
      msg = pylibtas.receiveMessage()

    utils.assert_equal(msg, pylibtas.MSGB_FRAMECOUNT_TIME)
    _, self.frame_counter = pylibtas.receiveULong()
    pylibtas.ignoreData(_SIZE_TIMESPEC)
    pylibtas.sendMessage(pylibtas.MSGN_EXPOSE)

    self.det.loadstate(index)
    trajectory, frame_buffer = self.saved_states[index]
    self.trajectory = trajectory.copy()
    self.frame_buffer = frame_buffer.copy()

  def _generate_goal_state(self):
    if np.random.uniform() < FLAGS.random_goal_prob and not GLOBAL.eval_mode:
      self.goal_y = np.random.randint(50, FLAGS.image_height - 50)
      self.goal_x = np.random.randint(50, FLAGS.image_width - 50)
    elif FLAGS.goal_y or FLAGS.goal_x:
      self.goal_y = FLAGS.goal_y
      self.goal_x = FLAGS.goal_x
    else:
      self.goal_y, self.goal_x = _GOAL_MAP[FLAGS.save_file]

  def reset(self):
    self.frame_buffer = []
    self.trajectory = []
    self._generate_goal_state()

  def frame_channels(self):
    return 4

  def extra_channels(self):
    return 1

  def num_actions(self):
    return len(self.class2button)

  def start_frame(self):
    msg = pylibtas.receiveMessage()
    while msg != pylibtas.MSGB_START_FRAMEBOUNDARY:
      if msg == pylibtas.MSGB_WINDOW_ID:
        pylibtas.ignoreData(_SIZE_INT)
      elif msg == pylibtas.MSGB_ALERT_MSG:
        logging.warning(pylibtas.receiveString())
      elif msg == pylibtas.MSGB_ENCODE_FAILED:
        raise RuntimeError('MSGB_ENCODE_FAILED')
      elif msg == pylibtas.MSGB_FRAMECOUNT_TIME:
        pylibtas.ignoreData(_SIZE_UNSIGNED_LONG)
        pylibtas.ignoreData(_SIZE_TIMESPEC)
      elif msg == pylibtas.MSGB_GAMEINFO:
        pylibtas.ignoreData(_SIZE_GAMEINFO_STRUCT)
      elif msg == pylibtas.MSGB_FPS:
        pylibtas.ignoreData(_SIZE_FLOAT)
        pylibtas.ignoreData(_SIZE_FLOAT)
      elif msg == pylibtas.MSGB_ENCODING_SEGMENT:
        pylibtas.ignoreData(_SIZE_INT)
      elif msg == pylibtas.MSGB_QUIT:
        raise RuntimeError('User Quit.')
      elif msg == -1:
        raise RuntimeError('The connection to the game was lost.')
      else:
        raise RuntimeError('Received unexpected message %s(%d)' % (pylibtas.message_name(msg), msg))
      msg = pylibtas.receiveMessage()
    pylibtas.sendMessage(pylibtas.MSGN_START_FRAMEBOUNDARY)

    if self.frame_counter == 0:
      pylibtas.sendMessage(pylibtas.MSGN_SAVESTATE_PATH)
      pylibtas.sendString('/tmp/celeste/savestate')

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

    action = None
    if self.moviefile and self.frame_counter < self.moviefile.nbFrames():
      action = pylibtas.AllInputs()
      action.emptyInputs()
      self.moviefile.getInputs(action, self.frame_counter)
    return frame, action

  def get_inputs_for_frame(self, frame):
    y, x, state = self.det.detect(frame, prior_coord=self.trajectory[-1] if self.trajectory else None)
    self.trajectory.append((y, x))

    window_shape = [FLAGS.image_height, FLAGS.image_width]
    # generate the full frame input by concatenating gaussian heat maps.
    if state == -1:
      gaussian_current_position = np.zeros(window_shape, dtype=np.float32)
    else:
      gaussian_current_position = utils.generate_gaussian_heat_map(window_shape, y, x)

    frame = frame.astype(np.float32).transpose([2, 0, 1]) / 255.0
    if hasattr(self, 'frame_buffer'):
      self.frame_buffer.append(frame)
    input_frame = torch.cat([torch.tensor(frame), torch.tensor(gaussian_current_position).unsqueeze(0)], 0)

    gaussian_goal_position = utils.generate_gaussian_heat_map(window_shape, self.goal_y, self.goal_x)
    extra_channels = torch.tensor(gaussian_goal_position).unsqueeze(0)

    input_frame = utils.downsample_image_to_input(input_frame)
    extra_channels = utils.downsample_image_to_input(extra_channels)

    if FLAGS.use_cuda:
      input_frame = input_frame.cuda()
      extra_channels = extra_channels.cuda()

    return input_frame, extra_channels

  def _rectangular_distance(self, y, x):
    return np.maximum(np.abs(y - self.goal_y), np.abs(x - self.goal_x))

  def get_reward(self):
    reward = 0
    should_end_episode = False

    y, x = self.trajectory[-1]
    if y is None:
      # Assume death
      should_end_episode = True
      reward -= 10
      y, x = self.trajectory[-2]

    # dist_to_goal = np.sqrt((y - self.goal_y)**2 + (x - self.goal_x)**2)
    dist_to_goal = self._rectangular_distance(y,x)
    # reward += 50 - 10 * dist_to_goal**0.33
    reward += -15 + 10*(float(dist_to_goal<450)) + 10*(float(dist_to_goal<250)) + 10*(float(dist_to_goal<50)) + 10*(float(dist_to_goal<5))

    if reward >= 48:
      should_end_episode = True
    return reward, should_end_episode

  def index_to_action(self, idx):
    action = pylibtas.AllInputs()
    action.emptyInputs()
    for button in self.class2button[idx]:
      if button not in _BUTTON_DICT:
        logging.warning('Unknown button %s!' % button)
        continue
      si = pylibtas.SingleInput()
      si.type = _BUTTON_DICT[button]
      action.setInput(si, 1)
    return action

  def end_frame(self, action):
    pylibtas.sendMessage(pylibtas.MSGN_ALL_INPUTS)
    pylibtas.sendAllInputs(action)
    pylibtas.sendMessage(pylibtas.MSGN_END_FRAMEBOUNDARY)
    self.frame_counter += 1

  def finish_episode(self, processed_frames):
    utils.assert_equal(len(self.frame_buffer), processed_frames + 1)
    utils.assert_equal(len(self.trajectory), processed_frames + 1)
    # utils.add_summary('video', 'input_frames', self.frame_buffer, fps=60)

    fig = plt.figure()
    plt.scatter(self.goal_x, self.goal_y, facecolors='none', edgecolors='r')
    utils.plot_trajectory(self.frame_buffer[-(self.det.death_clock_limit + 1)], self.trajectory)
    utils.add_summary('figure', 'trajectry', fig)

  def add_action_summaries(self, frame_number, softmax, sampled_idx):
    ax1_height_ratio = 3
    fig, (ax1, ax2) = plt.subplots(2, gridspec_kw={
        'height_ratios' : [ax1_height_ratio, 1],
    })
    ax1.scatter(self.goal_x, self.goal_y, facecolors='none', edgecolors='r')
    trajectory_i = self.trajectory[max(0, frame_number - FLAGS.context_frames):frame_number + 1]
    utils.plot_trajectory(self.frame_buffer[frame_number], trajectory_i, ax=ax1)
    ax1.axis('off')

    num_topk = 5
    topk_idxs = np.argsort(softmax)[::-1][:num_topk]
    labels = [','.join(self.class2button[idx]) for idx in topk_idxs]
    ax2.bar(np.arange(num_topk), softmax[topk_idxs], width=0.3)
    ax2.set_xticks(np.arange(num_topk))
    ax2.set_xticklabels(labels)
    ax2.set_ylim(0.0, 1.0)
    asp = np.diff(ax2.get_xlim())[0] / np.diff(ax2.get_ylim())[0]
    asp /= np.abs(np.diff(ax1.get_xlim())[0] / np.diff(ax1.get_ylim())[0])
    ax2.set_aspect(asp / ax1_height_ratio)
    ax2.set_title('Sampled: %s (%0.2f%%)' % (
        ','.join(self.class2button[sampled_idx]),
        softmax[sampled_idx] * 100.0))
    utils.add_summary('figure', 'action/frame_%03d' % frame_number, fig)
