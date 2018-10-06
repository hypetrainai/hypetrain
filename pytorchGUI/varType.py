class varType:
    def __init__(self, name, vt, default):
        self.name = name
        self.vt = vt
        self.default = default
    def __str__(self):
        return self.name + ": " + self.vt
    def __repr__(self):
        return str(self)