import imageio
import numpy as np
import os
from PIL import Image
import pylibtas
import queue
import signal
import sys
import torch.optim as optim
import torch
import torch.nn as nn

from celeste_detector import CelesteDetector
from model import ResNetIm2Value as Network
from GLOBALS import FLAGS, GLOBAL
from class2button import class2button

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


def startNextFrame():
  msg = pylibtas.receiveMessage()
  while msg != pylibtas.MSGB_START_FRAMEBOUNDARY:
    if msg == pylibtas.MSGB_WINDOW_ID:
      pylibtas.ignoreData(SIZE_INT)
    elif msg == pylibtas.MSGB_ALERT_MSG:
      print(pylibtas.receiveString())
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


def sample_action(scores):
    scores = scores.detach().cpu()
    dist = torch.distributions.categorical.Categorical(probs=scores)
    sample = dist.sample()
    sample_mapped = [class2button[int(sample[i].numpy())] for i in range(len(sample))]
    return sample, sample_mapped


class FrameProcessor(object):

  def __init__(self):
    self.det = CelesteDetector()
    self.prior_coord = None
    self.start_frame_saving = False
    self.saved_frames = 0

    self.goal = (440, 730)

    self.episode_number = 0
    self.episode_start = -1
    self.start_frame = None
    self.frame_buffer = None
    self.dist_to_goals = []
    self.sampled_action = []

    self.actor = nn.DataParallel(Network(FLAGS).cuda())
    self.critic = nn.DataParallel(Network(FLAGS, out_dim=1).cuda())
    self.optimizer_actor = optim.Adam(list(self.actor.parameters()), lr=FLAGS.lr)
    self.optimizer_critic = optim.Adam(list(self.critic.parameters()), lr=FLAGS.lr)

    if not os.path.isdir(FLAGS.log_dir):
        os.path.makedirs(FLAGS.log_dir)

    if FLAGS.pretrained_model_path:
        print('Loading pretrained model from %s' % FLAGS.pretrained_model_path)
        self.actor.load_state_dict(torch.load(
            FLAGS.pretrained_model_path + '/celeste_model_actor_%s.pt' % FLAGS.pretrained_suffix))
        self.critic.load_state_dict(torch.load(
            FLAGS.pretrained_model_path + '/celeste_model_critic_%s.pt' % FLAGS.pretrained_suffix))
        print('Done!')

  def finishEpisode(self):
    assert self.episode_start >= 0
    num_frames = frame_counter - self.episode_start
    assert len(self.dist_to_goals) == num_frames, (num_frames, len(self.dist_to_goals))
    assert len(self.sampled_action) == num_frames, (num_frames, len(self.sampled_action))
    assert self.frame_buffer.shape[1] == num_frames + FLAGS.context_frames, (num_frames, self.frame_buffer.shape[1])

    R = 0
    if self.dist_to_goals[-1] < 10**2:
      R = 100000

    last_V = None
    self.optimizer_actor.zero_grad()
    for i in reversed(range(num_frames)):
      frames = self.frame_buffer[:, i:i+FLAGS.context_frames]
      frames = torch.reshape(frames, [1, -1, FLAGS.image_height, FLAGS.image_width])
      V = self.critic.forward(frames)
      if not last_V:
        last_V = V.detach()
        continue

      R -= np.sqrt(self.dist_to_goals[i])

      self.optimizer_critic.zero_grad()
      ((R - V)**2).backward(retain_graph=True)
      self.optimizer_critic.step()

      last_V *= FLAGS.reward_decay_multiplier
      A = R + last_V - V
      scores = self.actor.forward(frames)
      (torch.log(scores[0, self.sampled_action[i]]) * A).backward()
      R *= FLAGS.reward_decay_multiplier
    self.optimizer_actor.step()

    # Start next episode.
    loadstate()
    self.episode_start = -1
    self.episode_number += 1
    if self.episode_number % FLAGS.save_every == 0:
        model_dir_actor = FLAGS.log_dir + '/celeste_model_actor_%d.pt' % self.episode_number
        model_dir_critic = FLAGS.log_dir + '/celeste_model_critic_%d.pt' % self.episode_number
        torch.save(self.actor.state_dict(), model_dir_actor)
        torch.save(self.critic.state_dict(), model_dir_critic)
        torch.save(self.actor.state_dict(), FLAGS.log_dir + '/celeste_model_actor_latest.pt')
        torch.save(self.critic.state_dict(), FLAGS.log_dir + '/celeste_model_critic_latest.pt')
    return self.processFrame(self.start_frame)

  def processFrame(self, frame):
    y, x, state = self.det.detect(frame, prior_coord=self.prior_coord)
    if y is not None:
      self.prior_coord = np.array([y, x]).astype(np.int)
      print('Character Location: (%f, %f), State: %d' % (y, x, state))
    else:
      self.prior_coord = None
      print('Character Location: Not Found! State: %d' % state)

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
      cuda_frame = torch.tensor(frame).float().permute(2, 0, 1).unsqueeze(0).cuda()
      if self.episode_start < 0:
        if self.start_frame is None:
          savestate()
        assert self.prior_coord is not None
        self.episode_start = frame_counter
        self.start_frame = frame
        self.frame_buffer = torch.stack([cuda_frame] * FLAGS.context_frames, 1)
        self.dist_to_goals = []
        self.sampled_action = []
      else:
        self.frame_buffer = torch.cat([self.frame_buffer, cuda_frame.unsqueeze(1)], 1)
        if self.prior_coord is None:
          # Assume death
          self.dist_to_goals.append(10000000)
          return self.finishEpisode()
        else:
          self.dist_to_goals.append((x - self.goal[1])**2 + (y - self.goal[0])**2)
        if frame_counter - self.episode_start >= FLAGS.episode_length:
          return self.finishEpisode()
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
      raise ValueError('Could not load movie %s' % sys.argv[1])
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
  shared_config.includeFlags = (
    pylibtas.LCF_ERROR |
    pylibtas.LCF_WARNING |
    pylibtas.LCF_CHECKPOINT)
  pylibtas.sendSharedConfig(shared_config)

  pylibtas.sendMessage(pylibtas.MSGN_ENCODING_SEGMENT)
  pylibtas.sendInt(0)

  pylibtas.sendMessage(pylibtas.MSGN_END_INIT)

  processor = FrameProcessor()
  action_queue = queue.Queue()
  while True:
    startNextFrame()

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
        for button_inputs in processor.processFrame(frame):
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


if __name__ == "__main__":
  try:
    Speedrun()
  except:
    if game_pid != -1:
      print('killing game %d' % game_pid)
      os.kill(game_pid, signal.SIGKILL)
    raise
