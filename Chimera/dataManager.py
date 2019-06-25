from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter.messagebox import showerror
import numpy as np
import tensorflow as tf
import ntpath

def load_file():
    fname = askopenfilename(filetypes=(("Numpy files", "*.npy"),
                                       ("All files", "*.*")))
    return fname


def load_inputs(input_l):
    fname = load_file()
    if fname:
        global x
        x = np.load(fname)
    text_to_display = ntpath.basename(fname) + " (" + str(x.shape[0]) + " entries)"
    input_l['text'] = text_to_display

def load_labels(labels_l):
    fname = load_file()
    if fname:
        global y
        y = np.load(fname)

        # if the array is one-dimensional, then reshape it to two-dimensional
        try:
            shape = y.shape[1]
        except IndexError:
            y = np.reshape(y, (-1, 1))
    text_to_display = ntpath.basename(fname) + " (" + str(y.shape[0]) + " entries)"
    labels_l['text'] = text_to_display


def prepare_data():
    if 'x' in globals() and 'y' in globals():
        assert x.shape[0] == y.shape[0]
        num_items = x.shape[0]

        inputShape = x.shape[1]
        outputShape = y.shape[1]

        # split the dataset into training and testing
        train = int(num_items * 0.7)
        train_examples = x[:train]
        train_labels = y[:train]
        test_examples = x[train:]
        test_labels = y[train:]

        return train_examples, train_labels, test_examples, test_labels
    else:
        showerror("Error", "You haven't selected data to train the model.\nMake sure you select the data by using Import > Import inputs and Import > Import labels")
        raise RuntimeError("data not selected")
