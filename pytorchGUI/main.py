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

root_canvas = tk.Canvas(root, width=canvas_width, height=canvas_height)
root_canvas.pack(expand = YES, fill = BOTH)
message = Label(root, text = "Press and Drag the mouse to draw")
message.pack(side = BOTTOM)

left_frame = tk.Frame(root_canvas, width=left_frame_width, height=canvas_height, highlightbackground="green", highlightthickness=1, bg="lavender")
root_canvas.create_window(0, 0, anchor=tk.NW, window=left_frame)

right_frame = tk.Canvas(root_canvas, width=1000,height=canvas_height)

root_canvas.create_window(left_frame_width, 0, anchor=tk.NW, window=right_frame)
main_canvas = right_frame

buttons = []
for i in range(0, len(layers)):
    layer_button = tk.Button(left_frame, text=layers[i].name, width=10, anchor=tk.NW)
    layer_button.grid(row=i, columnspan=2, sticky=tk.W)
    buttons.append(layer_button)

def Layer_Button_Action(i):
    layer = layers[i]
    for button in buttons:
        button.config(highlightbackground="white")
    buttons[i].config(highlightbackground="red")

    return Set_Current_Selected_Layer(layer)

for i in range(0, len(layers)):
    buttons[i].config(command=lambda i=i: Layer_Button_Action(i))

def Set_Current_Selected_Layer(layer):
    global selected_layer_type
    selected_layer_type = layer

all_nodes = []
node_names = NameCount()
list_other_widgets = []
selected_layer_type = None
network_settings = NetworkSettings()

disconnect_button = tk.Button(left_frame, text='Disconnect', width=10, anchor=tk.W)
disconnect_button.grid(row=len(layers)+len(list_other_widgets), columnspan=2, sticky=tk.W)
list_other_widgets.append(disconnect_button)

import_button = tk.Button(left_frame, text='Set Import Path', width=10, anchor=tk.W)
import_button.grid(row=len(layers)+len(list_other_widgets), columnspan=2, sticky=tk.W)
list_other_widgets.append(import_button)
import_button.config(command=network_settings.setPath)

create_code_button = tk.Button(left_frame, text='Create Code', width=10, anchor=tk.W)
create_code_button.grid(row=len(layers)+len(list_other_widgets), columnspan=2, sticky=tk.W)
list_other_widgets.append(create_code_button)
def Create_And_Run_Code():
    has_dupes, name = node_names.hasDupes()
    if has_dupes:
        print("ERROR - Duplicate layer name: " + name)
        return
    for button in buttons:
        button.config(highlightbackground="white")

    all_lines = GenerateCode(all_nodes)
    print("code is: ")
    print(all_lines)
    run_by_string(all_lines, network_settings)
create_code_button.config(command=lambda: Create_And_Run_Code())

left_frame_properties = tk.Frame(left_frame, bg="pale green")
left_frame_properties.grid(row=len(layers)+len(list_other_widgets))
selected_node = currLayer(left_frame_properties)

def Delete_Current_Node(event):
    if selected_node.layer is None:
        return
    if root.focus_get().winfo_class() == 'Entry':
        return
    node_names.removeName(selected_node.layer['name']);
    index = all_nodes.index(selected_node.layer)
    selected_node.delete(main_canvas)
    del(all_nodes[index])

def Disconnect_Current_Node():
    if selected_node.layer is None:
        return
    selected_node.disconnect(main_canvas)

disconnect_button.config(command=Disconnect_Current_Node)


def Select_Or_Create(event):
    global selected_layer_type
    if selected_layer_type is None:
        return
    x = event.x
    y = event.y
    selected = None
    for node in all_nodes:
        if (x - node.x)**2 + (y - node.y)**2 < 25**2:
            selected = node
            break
    if selected:
        selected_node.select(selected, main_canvas, node_names)
    else:
        selected = layer(x, y, main_canvas, selected_layer_type, node_names)
        selected_node.select(selected, main_canvas, node_names)
        all_nodes.append(selected)
        node_names.addName(selected['name'])
        
def Move_Node(event):
    selected = selected_node.layer
    if selected is None:
        return
    selected_node.move(selected, event.x, event.y, main_canvas)
    
def Connect_Nodes(event):
    if selected_node.layer is None:
        return
    x = event.x
    y = event.y
    selected = None
    for node in all_nodes:
        if (x - node.x)**2 + (y - node.y)**2 < 25**2:
            selected = node
            break
    if selected:
        selected.connect(selected_node.layer, main_canvas)
        selected_node.select(selected, main_canvas, node_names)

main_canvas.bind("<Button-1>", Select_Or_Create)
main_canvas.bind("<Button-3>", Connect_Nodes)
main_canvas.bind("<B1-Motion>", Move_Node)
root.bind("<BackSpace>", Delete_Current_Node)

mainloop()
