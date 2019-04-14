import os
import pylibtas

def Speedrun():
    print('Hello world!')
    pylibtas.launchGameThread(
        b'CelesteLinux/Celeste.bin.x86_64',
        b'libTAS/build64/libtas.so',
        b'',  # gameargs
        0,  # startframe
        b'CelesteLinux/lib64',
        os.path.dirname(os.path.abspath(__file__)).encode())
    os.wait()
    print('Goodbye world!')


if __name__ == "__main__":
    try:
        os.remove('/tmp/libTAS.socket')
    except:
        pass
    Speedrun()
