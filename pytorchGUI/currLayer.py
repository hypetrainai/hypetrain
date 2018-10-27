import tkinter as tk

class currLayer:
    def __init__(self, property_frame):
        self.layer = None
        self.rect  = None
        self.property_frame = property_frame
    def select(self, node, canvas):
        self.deselect(canvas)
        self.layer = node
        x1,y1 = (self.layer.x-25),(self.layer.y-25)
        x2,y2 = (self.layer.x+25),(self.layer.y+25)
        self.rect = canvas.create_rectangle(x1,y1, x2,y2)
        i = 0
        for layervar in self.layer.layervars:
            tk.Label(self.property_frame, text=layervar.vartype.name).grid(row=i)
            stringv = tk.StringVar()
            stringv.set(layervar.var)
            tk.Entry(self.property_frame, textvariable=stringv).grid(row=i, column=1)
            i = i + 1
            def entry_changed_callback(*args, stringv=stringv, layervar=layervar):
                layervar.var=stringv.get()
            stringv.trace("w", callback=entry_changed_callback)
    def deselect(self, canvas):
        if self.rect:
            canvas.delete(self.rect)
        self.rect = None
        for widget_to_destroy in self.property_frame.winfo_children():
            widget_to_destroy.destroy()
    def delete(self, canvas):
        curLayer = self.layer
        self.deselect(canvas)
        curLayer.delete(canvas)
