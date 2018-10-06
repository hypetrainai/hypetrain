from tkinter import *

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
        self.prev = []
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
                self.connect(prev[0], canvas, prev[1])
    def connect(self, prev, canvas, vec=None):
        if vec:
            canvas.delete(vec)
        if (prev,vec) in self.prev:
            self.prev.remove((prev,vec))
        vec = canvas.create_line(prev.x,prev.y, self.x, self.y, arrow=LAST)
        self.prev.append((prev,vec))
