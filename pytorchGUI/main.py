from tkinter import *
import tkinter as tk
from layerType import *
from varType import *
from layer import *
from currLayer import *
from NetworkSettings import *
from nnet_backend import *


layers = []
Input = layerType("Input", "snow", 1, 1)
Input.add_var("name", "str", 'Input')
Input.add_var("Image Size X", "int", 256)
Input.add_var("Image Size Y", "int", 256)
Input.add_var("Channel", "int", 3)
Conv_2D = layerType("Conv 2D", "blue", 1, 1)
Conv_2D.add_var("name", "str", 'conv2d')
Conv_2D.add_var("Filter Size", "int", 16)
Conv_2D.add_var("Filter Count", "int", 1)
Conv_2D.add_var("Stride", "int", 1)
Max_Pool_2D = layerType("Max Pool 2D", "green", 1, 1)
Max_Pool_2D.add_var("name", "str", 'maxpool')
Max_Pool_2D.add_var("Filter Size", "int", 16)
Max_Pool_2D.add_var("Stride", "int", 1)
Flatten = layerType("Flatten", "orange", 1, 1)
Flatten.add_var("name", "str", 'flatten')
Fully_Conn = layerType("Fully Conn", "red", 1, 1)
Fully_Conn.add_var("name", "str", 'FC')
Fully_Conn.add_var("hidden_size", "int", 64)

CE_loss = layerType("CE Loss", "purple", 1, 1)
CE_loss.add_var("name", "str", 'loss')

layers.append(Input)
layers.append(Conv_2D)
layers.append(Max_Pool_2D)
layers.append(Flatten)
layers.append(Fully_Conn)
layers.append(CE_loss)

canvas_width = 1280
canvas_height = 720
left_frame_width = 240
widget_height = 40

root = tk.Tk()
root.title("Draw the Network")

c = tk.Canvas(root, width=canvas_width, height=canvas_height)
c.pack(expand = YES, fill = BOTH)
message = Label( root, text = "Press and Drag the mouse to draw" )
message.pack( side = BOTTOM )

lf = tk.Frame(c, width=left_frame_width, height=canvas_height, highlightbackground="green", highlightthickness=1, bg="lavender")
c.create_window(0, 0, anchor=tk.NW, window=lf)

rf = tk.Canvas(c, width=1000,height=canvas_height)

c.create_window(left_frame_width, 0, anchor=tk.NW, window=rf)
main_canvas = rf

buttons = []
for i in range(0, len(layers)):
    b = tk.Button(lf, text=layers[i].name, width=10, anchor=tk.NW)
    b.grid(row=i, columnspan=2, sticky=tk.W)
    buttons.append(b)

def setin(i):
    inp = layers[i]
    for button in buttons:
        button.config(highlightbackground="white")
    buttons[i].config(highlightbackground="red")

    return setInput(inp)

for i in range(0, len(layers)):
    buttons[i].config(command=lambda i=i: setin(i))

def setInput(inp):
    global layertype
    layertype = inp

list_nodes = []
list_other_widgets = []
layertype = None
networkSettings = NetworkSettings()

delete_button = tk.Button(lf, text='Delete', width=10, anchor=tk.W)
delete_button.grid(row=len(layers)+len(list_other_widgets), columnspan=2, sticky=tk.W)
list_other_widgets.append(delete_button)

run_button = tk.Button(lf, text='Run', width=10, anchor=tk.W)
run_button.grid(row=len(layers)+len(list_other_widgets), columnspan=2, sticky=tk.W)
list_other_widgets.append(run_button)
def run_network():
    run_fx()
run_button.config(command=run_network)

import_button = tk.Button(lf, text='Set Import Path', width=10, anchor=tk.W)
import_button.grid(row=len(layers)+len(list_other_widgets), columnspan=2, sticky=tk.W)
list_other_widgets.append(import_button)
import_button.config(command=networkSettings.setPath)

b = tk.Button(lf, text='Create Code', width=10, anchor=tk.W)
b.grid(row=len(layers)+len(list_other_widgets), columnspan=2, sticky=tk.W)
list_other_widgets.append(b)

def setin2():
    for button in buttons:
        button.config(highlightbackground="white")
    #print('hello world')
    code_init = []
    code_forward = []
    code_init.append('class Test(nn.Module):')
    code_init.append('  def __init__(self):')
    code_init.append('    super(Test, self).__init__()')
    code_forward.append('  def forward(self,x):')
    for node in list_nodes:
        node.covered = False
    for node in list_nodes:
        if node.layer == layers[5]:
            code_init,code_forward = DFS_nodes(node, code_init,code_forward)
    for line in code_init:
        print(line)
    for line in code_forward:
        print(line)
    
def DFS_nodes(node, code_init,code_forward):
    node.covered = True
    for prev in node.prev:
        if not prev.covered:
            code_init,code_forward = DFS_nodes(prev, code_init,code_forward)
    if node.layer.name == 'Input' or node.layer.name == 'Flatten':
        return code_init,code_forward
    statement = '    ' + node.layervars[0].var + ' = '
    if node.layer.name == 'Conv 2D':
        padding = int((node.layervars[1].var-1)/2)
        statement += 'T.nn.Conv2d(in_channels=<INPUT>, out_channels=' + str(node.layervars[2].var) + ',kernel_size='+ str(node.layervars[1].var)+',padding='+str(padding)+')'
    elif node.layer.name == 'Max Pool 2D':
        statement += 'T.nn.MaxPool2d()'
    elif node.layer.name == 'Fully Conn':
        statement += 'T.nn.Linear()'
    elif node.layer.name == 'CE Loss':
        statement += 'T.nn.CrossEntropyLoss()'
    code_init.append(statement)
    return code_init,code_forward

b.config(command=lambda: setin2())

lf_properties = tk.Frame(lf, bg="pale green")
lf_properties.grid(row=len(layers)+len(list_other_widgets))
currNode = currLayer(lf_properties)

def delete_curr_node():
    if currNode.layer is None:
        return
    index = list_nodes.index(currNode.layer)
    currNode.delete(main_canvas)
    del(list_nodes[index])

delete_button.config(command=delete_curr_node)


def selectORcreate( event ):
    global layertype
    if layertype is None:
        return
    x = event.x
    y = event.y
    selected = None
    for node in list_nodes:
        if (x - node.x)**2 + (y - node.y)**2 < 25**2:
            selected = node
            break
    if selected:
        currNode.select(selected, main_canvas)
    else:
        selected = layer(x, y, main_canvas, layertype)
        currNode.select(selected, main_canvas)
        list_nodes.append(selected)
        
def move( event ):
    selected = currNode.layer
    if selected is None:
        return
    selected.move( event.x, event.y, main_canvas)
    for n in selected.nextLayers:
        n.move(n.x, n.y, main_canvas)
    currNode.select(selected, main_canvas)
    
def connect( event ):
    x = event.x
    y = event.y
    selected = None
    for node in list_nodes:
        if (x - node.x)**2 + (y - node.y)**2 < 25**2:
            selected = node
            break
    if selected:
        selected.connect(currNode.layer, main_canvas)
        currNode.select(selected, main_canvas)

main_canvas.bind( "<Button-1>", selectORcreate)
main_canvas.bind( "<Button-3>", connect)
main_canvas.bind( "<B1-Motion>", move)

mainloop()
