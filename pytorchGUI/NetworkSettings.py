from tkinter.filedialog import askdirectory

class NetworkSettings:
    def __init__(self):
        self.path = ""
    def setPath(self):
        oldPath = self.path
        self.path = askdirectory()
        print("previous path was " + oldPath + "\nnew path is " + self.path)

