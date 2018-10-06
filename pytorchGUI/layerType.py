from varType import *

class layerType:
    def __init__(self, name, color, inCount, outCount):
        self.name = name
        self.color = color
        self.inCount = inCount
        self.outCount = outCount
        self.vars = []
    def add_var(self, name, vt, default):
        self.vars.append(varType(name, vt, default))  