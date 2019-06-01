from Tkinter import *
from Model import Chimera
import numpy as np

def update_net():
    global master
    global layer_repr

    for label in layer_repr:
        label.grid_forget()

    layer_repr = []

    n_row = 2
    for l in nnet.layers:
        label = Label(master, text = l.layer_type, bg = '#fafafa')
        layer_repr.append(label)
        label.grid(row = n_row)
        n_row += 1

def add_layer(layer_type):
    global nnet
    nnet.add_layer(layer_type)

    update_net()

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

    modelname_e = Entry(master)
    modelname_e.grid(row = 0, column = 1)

    save_b = Button(master,
            text = "Save",
            state = DISABLED,
            command = nnet.save())
    save_b.grid(row = 0, column = 2)

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
