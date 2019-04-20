import os
import pylibtas

SIZE_INT = 4

def Speedrun():
    print('Hello world!')
    pylibtas.removeSocket()
    pylibtas.launchGameThread(
        b'CelesteLinux/Celeste.bin.x86_64',
        b'libTAS/build64/libtas.so',
        b'',  # gameargs
        0,  # startframe
        b'CelesteLinux/lib64',
        os.path.dirname(os.path.abspath(__file__)).encode())
    pylibtas.initSocketProgram()

    msg = pylibtas.receiveMessage()
    while msg != pylibtas.MSGB_END_INIT:
        print('Received message', msg)
        if msg == pylibtas.MSGB_PID:
            status, pid = pylibtas.receiveData(SIZE_INT)
            print('pid:', pid)

    pylibtas.sendMessage(pylibtas.MSGN_END_INIT)

    os.wait()
    print('Goodbye world!')


if __name__ == "__main__":
    Speedrun()
