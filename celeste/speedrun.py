import os
import pylibtas

SIZE_INT = 4
SIZE_FLOAT = 4
SIZE_UNSIGNED_LONG = 8
SIZE_TIMESPEC = 16
SIZE_GAMEINFO_STRUCT = 36
SIZE_WINDOW_STRUCT = 8

def startFrameMessages():
  msg = pylibtas.receiveMessage()
  while msg != pylibtas.MSGB_START_FRAMEBOUNDARY:
    if msg == pylibtas.MSGB_WINDOW_ID:
      pylibtas.ignoreData(SIZE_WINDOW_STRUCT)
    elif msg == pylibtas.MSGB_ALERT_MSG:
      print(pylibtas.receiveString())
    elif msg == pylibtas.MSGB_ENCODE_FAILED:
      return True
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
      return True
    if msg == -1:
      print("The connection to the game was lost.")
      return True
    msg = pylibtas.receiveMessage()
  pylibtas.sendMessage(pylibtas.MSGN_START_FRAMEBOUNDARY);
  return False


def Speedrun():
    print('Hello world!')
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
        status, pid = pylibtas.receiveInt()
      msg = pylibtas.receiveMessage()

    pylibtas.sendMessage(pylibtas.MSGN_CONFIG)
    shared_config = pylibtas.SharedConfig()
    shared_config.initial_framecount = 0
    shared_config.running = True
    shared_config.incremental_savestates = False
    shared_config.prevent_savefiles = False
    shared_config.recycle_threads = False
    shared_config.main_gettimes_threshold = [-1, -1, -1, 100, -1, -1];
    pylibtas.sendSharedConfig(shared_config)

    pylibtas.sendMessage(pylibtas.MSGN_ENCODING_SEGMENT)
    pylibtas.sendData(0, SIZE_INT)

    pylibtas.sendMessage(pylibtas.MSGN_END_INIT)

    while True:
      quit = startFrameMessages()
      if quit:
        break

    print('Goodbye world!')


if __name__ == "__main__":
    Speedrun()
