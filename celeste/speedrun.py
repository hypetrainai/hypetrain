import os
import signal
import numpy as np
from PIL import Image
import pylibtas

SIZE_INT = 4
SIZE_FLOAT = 4
SIZE_UNSIGNED_LONG = 8
SIZE_TIMESPEC = 16
SIZE_GAMEINFO_STRUCT = 36
SIZE_WINDOW_STRUCT = 8


game_pid = -1
window_width = 960
window_height = 540


def initializeSavestates():
  pylibtas.sendMessage(pylibtas.MSGN_SAVESTATE_PATH)
  pylibtas.sendString('/tmp/celeste_save')


def savestate(index=0):
  pylibtas.sendMessage(pylibtas.MSGN_SAVESTATE_INDEX)
  pylibtas.sendData(index, SIZE_INT)
  pylibtas.sendMessage(pylibtas.MSGN_SAVESTATE)


# TODO: move these functions all into a class so shared_config can be a class member.
def loadstate(shared_config, index=0):
  pylibtas.sendMessage(pylibtas.MSGN_SAVESTATE_INDEX)
  pylibtas.sendData(index, SIZE_INT)
  pylibtas.sendMessage(pylibtas.MSGN_LOADSTATE)

  msg = pylibtas.receiveMessage()
  if msg == pylibtas.MSGB_LOADING_SUCCEEDED:
    pylibtas.sendMessage(pylibtas.MSGN_CONFIG)
    pylibtas.sendSharedConfig(shared_config)

    msg = pylibtas.receiveMessage()

  assert msg == pylibtas.MSGB_FRAMECOUNT_TIME
  pylibtas.ignoreData(SIZE_UNSIGNED_LONG)
  pylibtas.ignoreData(SIZE_TIMESPEC)

  pylibtas.sendMessage(pylibtas.MSGN_EXPOSE)


def startNextFrame():
  msg = pylibtas.receiveMessage()
  while msg != pylibtas.MSGB_START_FRAMEBOUNDARY:
    if msg == pylibtas.MSGB_WINDOW_ID:
      pylibtas.ignoreData(SIZE_WINDOW_STRUCT)
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
    msg = pylibtas.receiveMessage()
  pylibtas.sendMessage(pylibtas.MSGN_START_FRAMEBOUNDARY)

button_Dict = {
    'a': pylibtas.SingleInput.IT_CONTROLLER1_BUTTON_A,
    'b': pylibtas.SingleInput.IT_CONTROLLER1_BUTTON_B,
    'x': pylibtas.SingleInput.IT_CONTROLLER1_BUTTON_X,
    'y': pylibtas.SingleInput.IT_CONTROLLER1_BUTTON_Y,
    'u': pylibtas.SingleInput.IT_CONTROLLER1_BUTTON_DPAD_UP,
    'd': pylibtas.SingleInput.IT_CONTROLLER1_BUTTON_DPAD_DOWN,
    'l': pylibtas.SingleInput.IT_CONTROLLER1_BUTTON_DPAD_LEFT,
    'r': pylibtas.SingleInput.IT_CONTROLLER1_BUTTON_DPAD_RIGHT,
    'rt': pylibtas.SingleInput.IT_CONTROLLER1_AXIS_TRIGGERRIGHT
}

def processFrame(prev_inputs, frame, input=None):
  new_inputs = pylibtas.AllInputs()
  new_inputs.emptyInputs()

  buttons_pushed = []

  for button in input:
      if button not in button_Dict:
        continue
      new_button = pylibtas.SingleInput()
      new_button.type = button_Dict[button]
      buttons_pushed.append(new_button)

  for button in buttons_pushed:
      new_inputs.setInput(button, 1)

  return new_inputs


def Speedrun():
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
    else:
      raise RuntimeError('Unexpected message %d in init!' % msg)
    msg = pylibtas.receiveMessage()

  pylibtas.sendMessage(pylibtas.MSGN_CONFIG)
  shared_config = pylibtas.SharedConfig()
  shared_config.nb_controllers = 1
  shared_config.incremental_savestates = False
  shared_config.savestates_in_ram = False
  shared_config.backtrack_savestate = False
  shared_config.prevent_savefiles = False
  shared_config.recycle_threads = True
  shared_config.write_savefiles_on_exit = False
  shared_config.main_gettimes_threshold = [-1, -1, -1, 100, -1, -1]
  shared_config.includeFlags = (
      pylibtas.LCF_ERROR |
      pylibtas.LCF_WARNING |
      pylibtas.LCF_INFO |
      pylibtas.LCF_CHECKPOINT)
  pylibtas.sendSharedConfig(shared_config)

  pylibtas.sendMessage(pylibtas.MSGN_ENCODING_SEGMENT)
  pylibtas.sendData(0, SIZE_INT)

  pylibtas.sendMessage(pylibtas.MSGN_END_INIT)

  ai = pylibtas.AllInputs()
  ai.emptyInputs()
  frame_counter = 0
  while True:
    startNextFrame()
    frame_counter += 1

    msg = pylibtas.receiveMessage()
    assert msg == pylibtas.MSGB_FRAME_DATA, msg
    _, actual_window_width = pylibtas.receiveInt()
    _, actual_window_height = pylibtas.receiveInt()
    assert actual_window_width == window_width and actual_window_height == window_height
    _, size = pylibtas.receiveInt()
    received, frame = pylibtas.receiveArray(size)
    assert received == size, (size, received)

    frame = np.reshape(frame, [window_height, window_width, 4])[:, :, :3]
    button_input = input('Buttons please! (comma separated)').split(',')
    ai = processFrame(ai, frame, input = button_input)

    if frame_counter == 1:
      initializeSavestates()
    if frame_counter == 100:
      savestate(0)
    if frame_counter > 100 and frame_counter % 500 == 0:
      loadstate(shared_config, 0)

    pylibtas.sendMessage(pylibtas.MSGN_ALL_INPUTS)
    pylibtas.sendAllInputs(ai)
    pylibtas.sendMessage(pylibtas.MSGN_END_FRAMEBOUNDARY)


if __name__ == "__main__":
  try:
    Speedrun()
  except:
    if game_pid != -1:
      print('killing game %d' % game_pid)
      os.kill(game_pid, signal.SIGKILL)
    raise
