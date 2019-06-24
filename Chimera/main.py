#!/usr/bin/python

from tkinter import *
from tkinter.messagebox import showerror
from Layers import Strato
import Model
from Model import Chimera
import numpy as np
from os import listdir, getcwd, mkdir
from os.path import isfile, join, isdir
from DataManager import load_inputs, load_labels, prepare_data
from ModelManager import load_model, select_export_filepath

global globalPath
globalPath = join(getcwd(), "Chimera")

global color
color = {
    "Dense": '#2196F3',
    "Conv1D": '#4CAF50',
    "Conv2D": '#009688',
    "Dropout": '#F44336'}


def switch_freeze(ix):
    strato = nnet.layers[ix]
    if strato.layer.trainable:
        strato.freeze()
    else:
        strato.unfreeze()
    update_net()


def move_up(ix):
    nnet.move_up(ix)
    update_net()


def move_down(ix):
    nnet.move_down(ix)
    update_net()


def remove(ix):
    nnet.remove_layer(ix)
    update_net()


def update_net():
    global layer_repr

    for label in layer_repr:
        label.grid_forget()

    layer_repr = []

    n_row = 5
    ix = 0

    for l in nnet.layers:
        frame = Frame(master,
                      bg=color[l.config['layer_type']],
                      padx=32,
                      pady=4,
                      width=60)
        frame.grid(row=n_row, column=1, columnspan=4, pady=2)
        layer_repr.append(frame)

        # label for layer type
        label = Label(frame, text=l.config['layer_type'],
                      bg=frame['bg'],
                      padx=4,
                      width=15,
                      font="Arial 14")
        label.grid(row=n_row, column=1)

        # label for layer shape
        shape_l = Label(frame, text=l.label,
                        bg=frame['bg'],
                        padx=4,
                        width=6,
                        font="Arial 14")
        shape_l.grid(row=n_row, column=2)

        # freeze/unfreeze button
        freeze_b_text = StringVar()
        if l.layer.trainable:
            freeze_b_text = "Freeze"
        else:
            freeze_b_text = "Unfreeze"

        freeze_b = Button(frame,
                          bg='#0D47A1',
                          foreground='#FFFFFF',
                          borderwidth=0,
                          text=freeze_b_text,
                          font="Arial 12",
                          command=lambda ix=ix: switch_freeze(ix))
        freeze_b.grid(row=n_row, column=3)

        if ix < len(nnet.layers) - 1:
            # button to move up
            if ix > 0:
                up_b = Button(frame,
                              image=upPhoto,
                              text="up",
                              command=lambda ix=ix: move_up(ix))
                up_b.grid(row=n_row, column=4)

            if ix < len(nnet.layers) - 2:
                # button to move down
                down_b = Button(frame,
                                image=downPhoto,
                                text="down",
                                command=lambda ix=ix: move_down(ix))
                down_b.grid(row=n_row, column=5)

            # button to remove
            remove_b = Button(frame,
                              image=removePhoto,
                              text="delete",
                              command=lambda ix=ix: remove(ix))
            remove_b.grid(row=n_row, column=6)

        n_row += 1
        ix += 1

    # set the selected item in loss menu and optimizer menu
    loss_choice.set(nnet.loss_name)
    opt_choice.set(nnet.optimizer_name)


def import_model():
    raise NotImplementedError()


def export():
    filepath = select_export_filepath()

    try:
        nnet.export(filepath)
    except RuntimeError:
        showerror("Error", "Could not build model.")


def add_layer(layer_type):
    """
    function to add a layer to the network
    args:
        layer_type (string):
    """
    nnet.add_layer(layer_type)
    # update the view of the network with the new layer
    update_net()


def update_list_models():
    menu = menu_models["menu"]
    menu.delete(0, "end")
    mypath = "models"
    models = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    for m in models:
        menu.add_command(label=m,
                         command=lambda value=m: model_choice.set(value))


def save_and_update():
    nnet.save(modelname_e.get())
    update_list_models()


def load_and_update(model_name):
    nnet.load(model_name)
    update_net()


def fit():
    try:
        train_examples, train_labels, test_examples, test_labels = prepare_data()
    except RuntimeError:
        print("data not selected")
        return

    history_obj = nnet.fit(train_examples, train_labels)
    accuracy = history_obj.history['accuracy'][-1]
    loss = history_obj.history['loss'][-1]
    accuracy_l['text'] = "Accuracy: " + str(accuracy)
    loss_l['text'] = "Loss: " + str(loss)
    update_net()


def create_window():
    global master, layer_repr

    # main window setup
    master = Tk()
    master.title("Chimera")
    master.minsize(1020, 620)
    master.geometry("1020x620")
    master['bg'] = '#fafafa'
    master.columnconfigure(0, weight=1)
    master.columnconfigure(1, weight=1)
    master.columnconfigure(2, weight=2)
    master.columnconfigure(3, weight=1)
    master.columnconfigure(4, weight=1)
    master.columnconfigure(5, weight=2)

    # all images
    global upPhoto, downPhoto, removePhoto
    imgDir = join(globalPath, "img")
    upPhoto = PhotoImage(file=join(imgDir, "arrow_up.png"))
    downPhoto = PhotoImage(file=join(imgDir, "arrow_down.png"))
    removePhoto = PhotoImage(file=join(imgDir, "delete.png"))
    savephoto = PhotoImage(file=join(imgDir, "save.png"))

    menubar = Menu(master)
    importMenu = Menu(menubar, tearoff=0)
    importMenu.add_command(label="Import Model", command=import_model)
    importMenu.add_command(label="Import Input", command=load_inputs)
    importMenu.add_command(label="Import Labels", command=load_labels)

    menubar.add_cascade(label="Import", menu=importMenu)
    menubar.add_command(label="Export", command=export)
    menubar.add_command(label="Train", command=fit)
    master.config(menu=menubar)

    model_l = Label(master, text="Model name:",
                    bg=master['bg'])
    model_l.grid(row=0, column=1, sticky="E", padx=4)

    # entry for model name
    global modelname_e
    modelname_e = Entry(master)
    modelname_e.grid(row=0, column=2, sticky="ew")

    # save button
    save_b = Button(master,
                    image=savephoto,
                    text="save",
                    command=save_and_update)
    save_b.grid(row=0, column=3, sticky="W", padx=4)

    # model menu
    load_model_l = Label(master, text="Saved models:",
                         bg=master['bg'])
    load_model_l.grid(row=1, column=1, sticky="E", padx=4)

    models = [f for f in listdir("models") if isfile(join("models", f))]
    if models == []:
        models.append("")
    global menu_models
    model_choice = StringVar(master)
    model_choice.set(models[0])
    menu_models = OptionMenu(master, model_choice, *models,
                             command=lambda model_name=model_choice.get(): load_and_update(model_name))
    menu_models.grid(row=1, column=2, sticky="ew",)

    # losses menu
    losses_l = Label(master, text="Losses:", bg=master['bg'])
    losses_l.grid(row=2, column=1, sticky="E", padx=4)

    global loss_choice
    losses = [i for i in Model.losses.keys()]
    loss_choice = StringVar(master)
    loss_choice.set('CategoricalCrossentropy')
    menu_losses = OptionMenu(master, loss_choice, *losses,
                             command=lambda loss_name=loss_choice.get(): nnet.set_loss(loss_name))
    menu_losses.grid(row=2, column=2, sticky="ew")

    # optimizers menu
    opt_l = Label(master, text="Optimizers:", bg=master['bg'])
    opt_l.grid(row=3, column=1, sticky="E", padx=4)

    global opt_choice
    optimizers = [i for i in Model.optimizers.keys()]
    opt_choice = StringVar(master)
    opt_choice.set('Adam')
    menu_opt = OptionMenu(master, opt_choice, *optimizers,
                          command=lambda opt_name=opt_choice.get(): nnet.set_optimizer(opt_name))
    menu_opt.grid(row=3, column=2, sticky="ew")

    # initialize empty list for layer repr
    layer_repr = []

    # buttons to add layers
    addDense_b = Button(master,
                        text="Add Dense",
                        width=15,
                        bg=color['Dense'],
                        command=lambda: add_layer("Dense"))
    addDense_b.grid(row=0, column=0)
    addConv1D_b = Button(master,
                         text="Add Conv1D",
                         width=15,
                         bg=color['Conv1D'],
                         command=lambda: add_layer("Conv1D"))
    addConv1D_b.grid(row=1, column=0)
    addConv2D_b = Button(master,
                         text="Add Conv2D",
                         width=15,
                         bg=color['Conv2D'],
                         command=lambda: add_layer("Conv2D"))
    addConv2D_b.grid(row=2, column=0)
    addDropout_b = Button(master,
                          text="Add Dropout",
                          width=15,
                          bg=color['Dropout'],
                          command=lambda: add_layer("Dropout"))
    addDropout_b.grid(row=3, column=0)

    global input_l, labels_l, accuracy_l, loss_l
    input_l = Label(master,
                    text="No input data",
                    font="Arial 11",
                    bg=master['bg'],
                    padx=16)
    input_l.grid(row=0, column=5, sticky="W")
    labels_l = Label(master,
                     text="No labels",
                     font="Arial 11",
                     bg=master['bg'],
                     padx=16)
    labels_l.grid(row=1, column=5, sticky="W")
    accuracy_l = Label(master,
                       text="No accuracy data ",
                       font="Arial 11",
                       bg=master['bg'],
                       padx=16)
    accuracy_l.grid(row=2, column=5, sticky="W")
    loss_l = Label(master,
                   text="No loss data",
                   font="Arial 11",
                   bg=master['bg'],
                   padx=16)
    loss_l.grid(row=3, column=5, sticky="W")

    update_net()
    mainloop()


def set_nnet():
    global nnet
    nnet = Chimera()
    nnet.add_layer("Dense")


def checkdir():
    if not isdir("models"):
        mkdir("models")
    if not isdir("layers"):
        mkdir("layers")


checkdir()
set_nnet()
create_window()
