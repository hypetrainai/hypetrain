from varType import *

class layerType:
    def __init__(self, behavior, color, prevCount, nextCount):
        self.behavior = behavior
        self.name = behavior.GetName()
        self.color = color
        # a prev or next count of -1 means as many connections as you want`
        self.prevCount = prevCount
        self.nextCount = nextCount
        self.vars = []

    def add_var(self, name, vt, default):
        self.vars.append(varType(name, vt, default))

    def validate_behavior(self):
        valid, error = self.behavior.Validate(self.vars)
        if not valid:
            print("Invalid layer " + self.name + " : " + error)