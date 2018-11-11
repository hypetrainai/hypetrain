class NameCount:
    def __init__(self):
        self.names = {}

    def addName(self, name):
        if name not in self.names:
            self.names[name] = 0
        self.names[name] += 1

    def removeName(self, name):
        if name in self.names:
            self.names[name] -= 1
            if self.names[name] <= 0:
                self.names.pop(name)

    def nextName(self, baseName):
        i = 1
        name = baseName + str(i)
        while name in self.names:
            i += 1
            name = baseName + str(i)
        return name

    def changeName(self, before, after):
        self.removeName(before)
        self.addName(after)
        return self.names[after]

    def hasDupes(self):
        for name, count in self.names.items():
            if count > 1:
                return True, name
        return False, None

    def count(self, name):
        if name not in self.names:
            return 0
        return self.names[name]