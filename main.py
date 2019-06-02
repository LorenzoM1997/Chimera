from Tkinter import *
from Layers import Strato
from Model import Chimera
import numpy as np
import os
from os import listdir
from os.path import isfile, join

def switch_freeze(ix):
    global nnet
    strato = nnet.layers[ix]
    if strato.layer.trainable:
        strato.freeze()
    else:
        strato.unfreeze()
    update_net()

def update_net():
    global master
    global layer_repr

    for label in layer_repr:
        label.grid_forget()

    layer_repr = []

    n_row = 2
    ix = 0
    for l in nnet.layers:
        frame = Frame(master,
                bg = '#2196F3',
                padx = 32,
                pady = 4)
        frame.grid(row = n_row, column= 2, columnspan = 3, pady = 2)
        layer_repr.append(frame)

        # label for layer type
        label = Label(frame, text = l.layer_type,
                bg = frame['bg'],
                padx = 4,
                font = "Arial 14")
        label.grid(row = n_row, column = 1)

        # label for layer shape
        shape_l = Label(frame, text = l.shape,
                bg = frame['bg'],
                padx = 8,
                font = "Arial 14")
        shape_l.grid(row = n_row, column = 2)

        # freeze/unfreeze button
        freeze_b_text = StringVar()
        if l.layer.trainable:
            freeze_b_text = "Freeze"
        else:
            freeze_b_text = "Unfreeze"

        freeze_b = Button(frame,
                bg = '#0D47A1',
                foreground = '#FFFFFF',
                borderwidth = 0,
                text = freeze_b_text,
                font = "Arial 12",
                command = lambda ix=ix: switch_freeze(ix),)
        freeze_b.grid(row = n_row, column = 3)

        n_row += 1
        ix += 1

def add_layer(layer_type):
    global nnet
    nnet.add_layer(layer_type)

    update_net()

def update_list_models():
    global menu_models
    global model_choice
    menu = menu_models["menu"]
    menu.delete(0, "end")
    mypath = "models"
    models = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    for m in models:
        menu.add_command(label = m,
                command=lambda value=m: model_choice.set(value))

def save_and_update():
    global nnet
    global modelname_e
    nnet.save(modelname_e.get())
    update_list_models()

def load_and_update():
    global nnet
    global model_choice
    nnet.load(model_choice.get())
    update_net()

def create_window():
    global nnet
    global x_train, y_train
    global master, layer_repr
    
    master = Tk()
    master.title("Chimera")
    master.minsize(960,720)
    master.geometry("960x720")
    master['bg'] = '#fafafa'
    master.columnconfigure(0, weight = 0)
    master.columnconfigure(1, weight = 0)
    master.columnconfigure(2, weight = 1)
    master.columnconfigure(3, weight = 1)
    master.columnconfigure(4, weight = 1)

    menubar = Menu(master)
    importMenu = Menu(menubar, tearoff = 0)
    importMenu.add_command(label = "Import Model")
    importMenu.add_command(label = "Import Data")
    menubar.add_cascade(label = "Import", menu = importMenu)
    menubar.add_command(label = "Train",
            command = lambda: nnet.fit(x_train, y_train))

    master.config(menu = menubar)

    model_l = Label(master, text = "Model name:",
            bg = master['bg'])
    model_l.grid(row = 0, column = 0, sticky = "E", padx = 4)

    # entry for model name
    global modelname_e
    modelname_e = Entry(master)
    modelname_e.grid(row = 0, column = 1, sticky = "W")

    # save button
    savephoto = PhotoImage(file="save.png")
    save_b = Button(master,
            image = savephoto,
            text = "save",
            command = save_and_update)
    save_b.grid(row = 0, column = 2, sticky = "W", padx = 4)

    # model menu
    mypath = "models"
    models = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    if models == []:
        models.append("")
    global menu_models, model_choice
    model_choice = StringVar(master)
    model_choice.set(models[0])
    menu_models = OptionMenu(master, model_choice, *models)
    menu_models.grid(row = 1, column = 1, sticky = "W")

    # load button
    load_b = Button(master,
            text = "load",
            command = load_and_update)
    load_b.grid(row = 1, column = 2, sticky = "W", padx = 4)

    # initialize empty list for layer repr
    layer_repr = []

    addDense_b = Button(master,
            text = "Add Dense",
            width = 15,
            command = lambda: add_layer("Dense"))
    addDense_b.grid (row = 2, column = 0)
    addConv1D_b = Button(master,
            text = "Add Conv1D",
            width = 15,
            command = lambda: add_layer("Conv1D"))
    addConv1D_b.grid(row = 3, column = 0)


    update_net()
    mainloop()

def set_data():
    global x_train, y_train
    x_train = np.random.rand(32, 5)
    y_train = np.random.rand(32, 2)

def set_nnet():
    global nnet
    nnet = Chimera()
    nnet.add_layer("Dense")

def checkdir():
    if not os.path.isdir("models"):
        os.mkdir("models")
    if not os.path.isdir("layers"):
        os.mkdir("layers")

checkdir()
set_data()
set_nnet()
create_window()
