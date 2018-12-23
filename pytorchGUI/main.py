from tkinter import *
import tkinter as tk
from layerType import *
from varType import *
from layer import *
from currLayer import *
from NetworkSettings import *
from nnet_backend import *
from name_count import *
from CodeGenerator import GenerateCode
import sys
import os
dirname = os.path.dirname(os.path.abspath(__file__)) + "/layers"
sys.path.append(dirname)
from input_layer import InputLayer
from conv2d_layer import Conv2DLayer
from max_pool2d_layer import MaxPool2DLayer
from flatten_layer import FlattenLayer
from fully_connected_layer import FullyConnectedLayer
from ce_loss_layer import CELossLayer

layers = []
Input = layerType(InputLayer(), "snow", 0, -1)
Input.add_var("name", "str", 'Input')
Input.add_var("Image Size X", "int", 256)
Input.add_var("Image Size Y", "int", 256)
Input.add_var("out_channels", "int", 3)
Input.validate_behavior()
Conv_2D = layerType(Conv2DLayer(), "blue", 1, 1)
Conv_2D.add_var("name", "str", 'conv2d')
Conv_2D.add_var("code", "str", 'torch.nn.Conv2d')
Conv_2D.add_var("out_channels", "int", 128)
Conv_2D.add_var("kernel_size", "int", 3)
Conv_2D.add_var("stride", "int", 1)
Conv_2D.add_var("Image Size X", "int", 256)
Conv_2D.add_var("Image Size Y", "int", 256)
Conv_2D.validate_behavior()
Max_Pool_2D = layerType(MaxPool2DLayer(), "green", 1, 1)
Max_Pool_2D.add_var("name", "str", 'maxpool')
Max_Pool_2D.add_var("code", "str", 'torch.nn.MaxPool2d')
Max_Pool_2D.add_var("kernel_size", "int", 3)
Max_Pool_2D.add_var("stride", "int", 2)
Max_Pool_2D.add_var("out_channels", "int", 128)
Max_Pool_2D.add_var("Image Size X", "int", 256)
Max_Pool_2D.add_var("Image Size Y", "int", 256)
Max_Pool_2D.validate_behavior()
Flatten = layerType(FlattenLayer(), "orange", 1, 1)
Flatten.add_var("name", "str", 'flatten')
Flatten.add_var("out_features", "int", 32768)
Flatten.validate_behavior()
Fully_Conn = layerType(FullyConnectedLayer(), "red", 1, 1)
Fully_Conn.add_var("name", "str", 'FC')
Fully_Conn.add_var("code", "str", 'torch.nn.Linear')
Fully_Conn.add_var("out_features", "int", 128)
Fully_Conn.validate_behavior()
CE_loss = layerType(CELossLayer(), "purple", 2, 0)
CE_loss.add_var("name", "str", 'loss')
CE_loss.add_var("code", "str", 'torch.nn.CrossEntropyLoss')
CE_loss.validate_behavior()

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
node_names = NameCount()
list_other_widgets = []
layertype = None
networkSettings = NetworkSettings()

disconnect_button = tk.Button(lf, text='Disconnect', width=10, anchor=tk.W)
disconnect_button.grid(row=len(layers)+len(list_other_widgets), columnspan=2, sticky=tk.W)
list_other_widgets.append(disconnect_button)

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
    hasDupes, name = node_names.hasDupes()
    if hasDupes:
        print("ERROR - Duplicate layer name: " + name)
        return
    for button in buttons:
        button.config(highlightbackground="white")

    all_lines = GenerateCode(list_nodes)
    print("code is: ")
    print(all_lines)
    run_by_string(all_lines, networkSettings)

b.config(command=lambda: setin2())

lf_properties = tk.Frame(lf, bg="pale green")
lf_properties.grid(row=len(layers)+len(list_other_widgets))
currNode = currLayer(lf_properties)

def delete_curr_node(event):
    if currNode.layer is None:
        return
    if root.focus_get().winfo_class() == 'Entry':
        return
    node_names.removeName(currNode.layer['name']);
    index = list_nodes.index(currNode.layer)
    currNode.delete(main_canvas)
    del(list_nodes[index])

def disconnect_curr_node():
    if currNode.layer is None:
        return
    currNode.disconnect(main_canvas)

disconnect_button.config(command=disconnect_curr_node)


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
        currNode.select(selected, main_canvas, node_names)
    else:
        selected = layer(x, y, main_canvas, layertype, node_names)
        currNode.select(selected, main_canvas, node_names)
        list_nodes.append(selected)
        node_names.addName(selected['name'])
        
def move( event ):
    selected = currNode.layer
    if selected is None:
        return
    currNode.move(selected, event.x, event.y, main_canvas)
    
def connect( event ):
    if currNode.layer is None:
        return
    x = event.x
    y = event.y
    selected = None
    for node in list_nodes:
        if (x - node.x)**2 + (y - node.y)**2 < 25**2:
            selected = node
            break
    if selected:
        selected.connect(currNode.layer, main_canvas)
        currNode.select(selected, main_canvas, node_names)

main_canvas.bind( "<Button-1>", selectORcreate)
main_canvas.bind( "<Button-3>", connect)
main_canvas.bind( "<B1-Motion>", move)
root.bind( "<BackSpace>", delete_curr_node)

mainloop()
