from tkinter import *
from tkinter.filedialog import askopenfilename, asksaveasfilename
from os.path import join
from os import getcwd


def load_model():
    fname = askopenfilename(filetypes=(("Keras models", "*.h5"),
                                       ("All files", "*.*")))
    return fname


def select_export_filepath():
    modelDir = join(getcwd(), "models")
    fname = asksaveasfilename(initialdir=modelDir,
                              title="Export file",
                              filetypes=(("Keras models", "*.h5"),
                                         ("All files", "*.*")))
    return fname
