from tkinter import *
import traceback

class layerVar:
    def __init__(self, vartype):
        self.vartype = vartype
        self.var = vartype.default
        
class layer:
    def __init__(self, x, y, canvas, layertype):
        self.x = x
        self.y = y
        self.layer = layertype
        self.w = self.create_oval(canvas)
        self.prev = {}
        self.nextLayers = {}
        self.layervars = []
        for varT in layertype.vars:
            self.layervars.append(layerVar(varT))
    def create_oval(self, canvas):
        x1,y1 = (self.x-25), (self.y-25)
        x2,y2 = (self.x+25), (self.y+25)                              
        return canvas.create_oval(x1, y1, x2, y2, fill=self.layer.color)
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

