from Tkinter import *
from Layers import Strato
from Model import Chimera
import numpy as np
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
        frame = Frame(master, bg = '#fafafa')
        frame.grid(row = n_row)
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

def create_window():
    global nnet
    global x_train, y_train
    global master, layer_repr
    
    master = Tk()
    master.title("Chimera")
    master.minsize(600,400)
    master.geometry("600x400")
    master['bg'] = '#fafafa'

    train_button = Button(master,
            text="Train network",
            command = lambda: nnet.fit(x_train, y_train))
    train_button.grid(row = 0, column = 0)

    addDense_b = Button(master,
            text = "Add Dense",
            command = lambda: add_layer("Dense"))
    addDense_b.grid (row = 1)

    global modelname_e
    modelname_e = Entry(master)
    modelname_e.grid(row = 0, column = 1)

    save_b = Button(master,
            text = "Save",
            command = save_and_update)
    save_b.grid(row = 0, column = 2)

    mypath = "models"
    models = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    global menu_models, model_choice
    model_choice = StringVar(master)
    model_choice.set(models[0])
    menu_models = OptionMenu(master, model_choice, *models)
    menu_models.grid(row = 0, column = 3)

    load_b = Button(master,
            text = "load",
            command = lambda: nnet.load(model_choice.get()))
    load_b.grid(row = 0, column = 4)

    # initialize empty list for layer repr
    layer_repr = []

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
    nnet.add_layer("Dense")

set_data()
set_nnet()
create_window()
