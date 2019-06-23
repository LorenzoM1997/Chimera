from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter.messagebox import showerror
import numpy as np
import tensorflow as tf

def load_file():
    fname = askopenfilename(filetypes=(("Numpy files", "*.npy"),
                                       ("All files", "*.*")))
    return fname

def load_inputs():
    fname = load_file()
    if fname:
        global x
        x = np.load(fname)


def load_labels():
    fname = load_file()
    if fname:
        global y
        y = np.load(fname)


def prepare_data():
    if 'x' in globals() and 'y' in globals():
        assert x.shape[0] == y.shape[0]
        num_items = x.shape[0]

        inputShape = x.shape[1]
        outputShape = y.shape[1]

        # split the dataset into training and testing
        train = int(num_items * 0.7)
        train_examples, train_labels = x[:train], y[:train]
        test_examples, test_labels = x[train:], y[:train]

        # convert the tensors into Datasets
        global train_dataset, test_dataset
        train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))
        test_dataset = tf.data.Dataset.from_tensor_slices((test_examples, test_labels))

        return train_dataset, inputShape, outputShape