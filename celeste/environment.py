from absl import flags
import numpy as np
import os

import pylibtas
import utils

FLAGS = flags.FLAGS

_SIZE_INT = 4
_SIZE_FLOAT = 4
_SIZE_UNSIGNED_LONG = 8
_SIZE_TIMESPEC = 16
_SIZE_GAMEINFO_STRUCT = 36


class Environment(object):

  def __init__(self):
    self.frame_counter = 0

    os.system('mkdir -p /tmp/celeste/movies')
    os.system('cp -f settings.celeste ~/.local/share/Celeste/Saves/')
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

  def savestate(self, index):
    pylibtas.sendMessage(pylibtas.MSGN_SAVESTATE_INDEX)
    pylibtas.sendInt(index)
    pylibtas.sendMessage(pylibtas.MSGN_SAVESTATE)
    utils.assert_equal(pylibtas.receiveMessage(), pylibtas.MSGB_SAVING_SUCCEEDED)

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
    return frame

  def end_frame(self, all_inputs):
    pylibtas.sendMessage(pylibtas.MSGN_ALL_INPUTS)
    pylibtas.sendAllInputs(all_inputs)
    pylibtas.sendMessage(pylibtas.MSGN_END_FRAMEBOUNDARY)
    self.frame_counter += 1
