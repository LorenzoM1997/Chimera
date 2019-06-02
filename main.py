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
        frame.grid(row = n_row, columnspan = 3, pady = 2)
        layer_repr.append(frame)

        label = Label(frame, text = l.layer_type, bg = frame['bg'])
        label.grid(row = n_row, column = 1)

        freeze_b_text = StringVar()
        if l.layer.trainable:
            freeze_b_text = "Freeze"
        else:
            freeze_b_text = "Unfreeze"

        freeze_b = Button(frame, text = freeze_b_text, command = lambda ix=ix: switch_freeze(ix))
        freeze_b.grid(row = n_row, column = 2)

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

    menubar = Menu(master)
    importMenu = Menu(menubar, tearoff = 0)
    importMenu.add_command(label = "Import Model")
    importMenu.add_command(label = "Import Data")
    menubar.add_cascade(label = "Import", menu = importMenu)
    menubar.add_command(label = "Train",
            command = lambda: nnet.fit(x_train, y_train))

    master.config(menu = menubar)

    global modelname_e
    modelname_e = Entry(master)
    modelname_e.grid(row = 0, column = 1)

    savephoto = PhotoImage(file="save.png")

    save_b = Button(master,
            image = savephoto,
            text = "save",
            command = save_and_update)
    save_b.grid(row = 0, column = 2)

    mypath = "models"
    models = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    if models == []:
        models.append("")
    global menu_models, model_choice
    model_choice = StringVar(master)
    model_choice.set(models[0])
    menu_models = OptionMenu(master, model_choice, *models)
    menu_models.grid(row = 0, column = 3)

    load_b = Button(master,
            text = "load",
            command = load_and_update)
    load_b.grid(row = 0, column = 4)

    # initialize empty list for layer repr
    layer_repr = []

    addDense_b = Button(master,
            text = "Add Dense",
            command = lambda: add_layer("Dense"))
    addDense_b.grid (row = 1)

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
