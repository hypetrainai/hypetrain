from absl import flags
from absl import logging
import inspect
import matplotlib.pyplot as plt
import numpy as np
import os
import signal
import torch
from torch.nn import functional as F

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
    super().__init__()
    assert FLAGS.batch_size == 1

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
    self.saved_states[index] = self.trajectory[-1:]

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
    self.trajectory = self.saved_states[index].copy()

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
    self.trajectory = []
    self._generate_goal_state()
    self.prev_done = False
    self.prev_reward = None
    self.min_reward = None

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
      pylibtas.sendString(FLAGS.savestate_path)

    msg = pylibtas.receiveMessage()
    utils.assert_equal(msg, pylibtas.MSGB_FRAME_DATA)
    _, actual_window_width = pylibtas.receiveInt()
    utils.assert_equal(actual_window_width, FLAGS.image_width)
    _, actual_window_height = pylibtas.receiveInt()
    utils.assert_equal(actual_window_height, FLAGS.image_height)
    _, size = pylibtas.receiveInt()
    received, frame = pylibtas.receiveArray(size)
    utils.assert_equal(received, size)

    frame = np.reshape(frame, [1, FLAGS.image_height, FLAGS.image_width, 4])[..., :3]
    frame = frame.transpose([0, 3, 1, 2])

    actions = None
    if self.moviefile and self.frame_counter < self.moviefile.nbFrames():
      action = pylibtas.AllInputs()
      action.emptyInputs()
      self.moviefile.getInputs(action, self.frame_counter)
      actions = [action]
    return frame, actions

  def get_inputs_for_frame(self, frame):
    frame = frame[0]
    y, x, state = self.det.detect(frame, prior_coord=self.trajectory[-1] if self.trajectory else None)
    self.trajectory.append((y, x))

    window_shape = [FLAGS.image_height, FLAGS.image_width]
    # generate the full frame input by concatenating gaussian heat maps.
    if state == -1:
      gaussian_current_position = np.zeros(window_shape, dtype=np.float32)
    else:
      gaussian_current_position = utils.generate_gaussian_heat_map(window_shape, y, x)

    frame = frame.astype(np.float32) / 255.0
    input_frame = torch.cat([utils.to_tensor(frame), utils.to_tensor(gaussian_current_position).unsqueeze(0)], 0)

    gaussian_goal_position = utils.generate_gaussian_heat_map(window_shape, self.goal_y, self.goal_x)
    extra_channels = utils.to_tensor(gaussian_goal_position).unsqueeze(0)

    input_frame = utils.downsample_image_to_input(input_frame.unsqueeze(0))
    extra_channels = utils.downsample_image_to_input(extra_channels.unsqueeze(0))

    return input_frame, extra_channels

  def _rectangular_distance(self, y, x):
    return np.maximum(np.abs(y - self.goal_y), np.abs(x - self.goal_x))

  def get_differential_reward(self, curr_reward):
    prev_reward = self.prev_reward
    if not prev_reward:
      self.prev_reward = curr_reward
      self.min_reward = curr_reward
      return 0.0

    if curr_reward > self.min_reward and curr_reward > prev_reward:
      reward = curr_reward - prev_reward
    else:
      reward = 0.0

    self.prev_reward = curr_reward
    self.min_reward = np.maximum(self.min_reward, curr_reward)

    return reward

  def dist_reward(self, x, y):
    return 50 - 10 * (np.sqrt((y - self.goal_y)**2 + (x - self.goal_x)**2))**0.33

  def get_reward(self):
    reward = 0
    final_reward = 0
    done = self.prev_done

    y, x = self.trajectory[-1]
    if y is None:
      # Assume death
      done = True
      final_reward -= 50.0*10
      y, x = self.trajectory[-2]

    curr_reward = 50.0*self.dist_reward(x, y)
    if FLAGS.differential_reward:
        final_reward += self.get_differential_reward(curr_reward)
    else:
        final_reward += curr_reward
    #dist_to_goal = self._rectangular_distance(y,x)
    #reward += -15 + 10*(float(dist_to_goal<450)) + 10*(float(dist_to_goal<250)) + 10*(float(dist_to_goal<50)) + 10*(float(dist_to_goal<5))

    if curr_reward >= 40 * 50.0:
      done = True

    self.prev_done = done
    return np.array([final_reward]), np.array([done])

  def indices_to_actions(self, idxs):
    actions = []
    for i in range(len(idxs)):
      action = pylibtas.AllInputs()
      action.emptyInputs()
      for button in self.class2button[idxs[i]]:
        if button not in _BUTTON_DICT:
          logging.warning('Unknown button %s!' % button)
          continue
        si = pylibtas.SingleInput()
        si.type = _BUTTON_DICT[button]
        action.setInput(si, 1)
      actions.append(action)
    return actions

  def indices_to_labels(self, idxs):
    return [','.join(self.class2button[idxs[i]]) for i in range(len(idxs))]

  def end_frame(self, actions):
    pylibtas.sendMessage(pylibtas.MSGN_ALL_INPUTS)
    pylibtas.sendAllInputs(actions[0])
    pylibtas.sendMessage(pylibtas.MSGN_END_FRAMEBOUNDARY)
    self.frame_counter += 1

  def finish_episode(self, processed_frames, frame_buffer):
    utils.assert_equal(len(frame_buffer), processed_frames + 1)
    utils.assert_equal(len(self.trajectory), processed_frames + 1)
    self.prev_reward = None
    self.min_reward = None

    fig = plt.figure()
    plt.scatter(self.goal_x, self.goal_y, facecolors='none', edgecolors='r')
    utils.plot_trajectory(self.trajectory, frame_buffer[-(self.det.death_clock_limit + 1)][0])
    utils.add_summary('figure', 'trajectory', fig)

  def _add_action_summaries_image(self, ax, frame_number, frame):
    super()._add_action_summaries_image(ax, frame_number, frame)
    trajectory_i = self.trajectory[max(0, frame_number - FLAGS.action_summary_frames):frame_number + 1]
    utils.plot_trajectory(trajectory_i, ax=ax)
    ax.scatter(self.goal_x, self.goal_y, facecolors='none', edgecolors='r')
