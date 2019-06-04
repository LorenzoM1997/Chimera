import os
import random
import string
import tensorflow as tf
import numpy as np
import pickle

from tensorflow.keras.layers import Layer, Dense, Flatten, Conv2D, Conv1D


def check_dir():
    if not os.path.isdir("layers"):
        os.mkdir("layers")


class Strato(object):
    def __init__(self, name=None, layer_type="Dense"):
        self.layer_type = layer_type

        self.units = 16  # default in case that name is not declared
        self.filters = 16  # default in case that name is not declared
        self.assemble()

        self.frozen = False

        check_dir()

        if name == None:
            self.weights = None
            self.name = ''.join(random.choice(
                string.ascii_lowercase + string.digits) for _ in range(16))

            self.path = os.path.join("layers", self.name)
            while os.path.exists(self.path):
                self.name = ''.join(random.choice(
                    string.ascii_lowercase + string.digits) for _ in range(16))
                self.path = os.path.join("layers", self.name)
        else:
            self.name = name
            self.load()

    def assemble(self):
        if self.layer_type == "Dense":
            self.layer = Dense(
                units=self.units,
                activation='relu')
            self.shape = self.units
        elif self.layer_type == "Conv1D":
            self.layer = Conv1D(
                filters=self.filters,
                kernel_size=5,
                strides=2,
                padding="same",
                activation="relu")
            self.shape = self.filters
        else:
            raise ValueError("Incorrect layer type")

    def freeze(self):
        self.layer.trainable = False
        self.frozen = True

    def unfreeze(self):
        self.layer.trainable = True
        self.frozen = False

    def save(self):
        weights = self.layer.get_weights()
        check_dir()
        filename = os.path.join("layers", self.name)
        wname = filename + ".weights"

        if os.path.exists(wname):
            os.remove(wname)
        pickle.dump(weights, open(wname, "wb"))

        info = {'layer_type': self.layer_type,
                'frozen': self.frozen,
                'input_shape': self.layer.input_shape}
        iname = filename + ".info"
        if os.path.exists(iname):
            os.remove(iname)
        pickle.dump(info, open(iname, "wb"))

    def load(self):
        check_dir()
        filename = os.path.join("layers", self.name)
        wname = filename + ".weights"
        self.weights = pickle.load(open(wname, "rb"))

        iname = filename + ".info"
        info = pickle.load(open(iname, "rb"))
        self.layer_type = info['layer_type']
        self.frozen = info['frozen']
        if self.layer_type == "Dense":
            self.units = self.weights[0].shape[1]

        self.assemble()
        
    def resetWeights(self):
        raise NotImplemented()
