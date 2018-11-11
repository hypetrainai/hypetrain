from tkinter import *
import traceback
import sys

class layerVar:
    def __init__(self, vartype):
        self.vartype = vartype
        self.var = vartype.default
        
class layer:
    def __init__(self, x, y, canvas, layertype, nameCount):
        self.x = x
        self.y = y
        self.type = layertype
        self.w = self.create_oval(canvas)
        self.prev = {}
        self.nextLayers = {}
        self.layervars = []
        for varT in layertype.vars:
            self.layervars.append(layerVar(varT))
        self['name'] = nameCount.nextName(self['name'])

    def create_oval(self, canvas):
        x1,y1 = (self.x-25), (self.y-25)
        x2,y2 = (self.x+25), (self.y+25)                              
        return canvas.create_oval(x1, y1, x2, y2, fill=self.type.color)
    def move(self, x, y, canvas):
        self.x = x
        self.y = y
        canvas.delete(self.w)
        self.w = self.create_oval(canvas)
        if self.prev:
            for prev in self.prev:
                self.connect(prev, canvas)
    def connect(self, prev, canvas):
        if prev in self.prev:
            canvas.delete(self.prev[prev])
        elif self.prev_available() <= 0 or prev.next_available() <= 0:
            return
        line = canvas.create_line(prev.x, prev.y, self.x, self.y, arrow=LAST)
        self.prev[prev] = line
        prev.nextLayers[self] = line

    def delete(self, canvas):
        canvas.delete(self.w)
        for prev in self.prev:
            canvas.delete(self.prev[prev])
            del(prev.nextLayers[self])
        for n in self.nextLayers:
            canvas.delete(self.nextLayers[n])
            del(n.prev[self])

        self.prev = None
        self.nextLayers = None

    def disconnect_next(self, canvas):
        for n in self.nextLayers:
            canvas.delete(self.nextLayers[n])
            del(n.prev[self])
        self.nextLayers = {}

    def prev_available(self):
        if self.type.prevCount == -1:
            return sys.maxsize
        return self.type.prevCount - len(self.prev)

    def next_available(self):
        if self.type.nextCount == -1:
            return sys.maxsize
        return self.type.nextCount - len(self.nextLayers)

    def get_vars(self):
        var_array = {}
        for layervar in self.layervars:
            var_array[layervar.vartype.name] = layervar.var
        return var_array

    def __getitem__(self, key):
        for layervar in self.layervars:
            if layervar.vartype.name == key:
                return layervar.var
        print("could not find key " + key)
        return None

    def __setitem__(self, key, item):
        for layervar in self.layervars:
            if layervar.vartype.name == key:
                layervar.var = item
                return;
        print("could not find key " + key)

