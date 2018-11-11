import tkinter as tk
import traceback

class currLayer:
    def __init__(self, property_frame):
        self.layer = None
        self.rect  = None
        self.property_frame = property_frame
    def select(self, node, canvas, nameCount):
        self.deselect(canvas)
        self.layer = node
        self.generateRect(self.layer, canvas)
        i = 0
        for layervar in self.layer.layervars:
            tk.Label(self.property_frame, text=layervar.vartype.name).grid(row=i)
            stringv = tk.StringVar()
            stringv.set(layervar.var)
            entry = tk.Entry(self.property_frame, textvariable=stringv)
            entry.grid(row=i, column=1)
            self.set_entry_bg(entry, layervar, nameCount)
            i = i + 1
            def entry_changed_callback(*args, stringv=stringv, layervar=layervar, nameCount=nameCount, entry=entry):
                nameCount.changeName(layervar.var, stringv.get())
                layervar.var = stringv.get()
                self.set_entry_bg(entry, layervar, nameCount)
            stringv.trace("w", callback=entry_changed_callback)

    def deselect(self, canvas):
        if self.rect:
            canvas.delete(self.rect)
        self.rect = None
        for widget_to_destroy in self.property_frame.winfo_children():
            widget_to_destroy.destroy()
    def move(self, node, x, y, canvas):
        if self.layer != node:
            self.select(node, canvas)
        if self.rect:
            canvas.delete(self.rect)
        self.layer.move(x, y, canvas)
        for n in self.layer.nextLayers:
            n.move(n.x, n.y, canvas)
        self.generateRect(self.layer, canvas)

    def generateRect(self, node, canvas):
        x1, y1 = (node.x - 25), (node.y - 25)
        x2, y2 = (node.x + 25), (node.y + 25)
        self.rect = canvas.create_rectangle(x1, y1, x2, y2)

    def delete(self, canvas):
        curLayer = self.layer
        self.deselect(canvas)
        curLayer.delete(canvas)
        self.layer = None

    def disconnect(self, canvas):
        self.layer.disconnect_next(canvas)

    def set_entry_bg(self, entry, layervar, nameCount):
        if layervar.vartype.name == 'name':
            if nameCount.count(layervar.var) > 1:
                entry.config({"bg": "Red"})
            else:
                entry.config({"bg": "White"})

