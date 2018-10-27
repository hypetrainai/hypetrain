from varType import *

class layerType:
    def __init__(self, name, color, prevCount, nextCount):
        self.name = name
        self.color = color
        # a prev or next count of -1 means as many connections as you want`
        self.prevCount = prevCount
        self.nextCount = nextCount
        self.vars = []
    def add_var(self, name, vt, default):
        self.vars.append(varType(name, vt, default))  