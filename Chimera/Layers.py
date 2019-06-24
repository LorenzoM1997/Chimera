from os import remove, mkdir
from os.path import join, exists, isdir
import random
import string
import tensorflow as tf
import numpy as np
import pickle

from tensorflow.keras.layers import Layer, Dense, Flatten, Conv2D, Conv1D, Dropout


def check_dir():
    if not isdir("layers"):
        mkdir("layers")


class Strato(object):
    def __init__(self, name=None, config=None):

        # create directory if not existent
        check_dir()

        # default settings
        self.config = {
            'layer_type': 'Dense',
            'frozen': False,
            'activation': "relu",
            'dense_units': 16,
            'filters': 8,
            'kernel_size': 5,
            'strides': 2,
            'padding': "same",
            'Dropout_rate': 0.2}

        # if there are configurations, use them to overwrite the default settings
        if config != None:
            for k in config.keys():
                self.config[k] = config[k]

        # assemble the layer
        self.assemble()

        if name == None:
            self.weights = None
            self.name = ''.join(random.choice(
                string.ascii_lowercase + string.digits) for _ in range(16))

            self.path = join("layers", self.name)
            while exists(self.path):
                self.name = ''.join(random.choice(
                    string.ascii_lowercase + string.digits) for _ in range(16))
                self.path = join("layers", self.name)
        else:
            self.name = name
            self.path = join("layers", self.name)
            self.load()

    def assemble(self):

        if self.config['layer_type'] == "Dense":
            self.layer = Dense(
                units=self.config['dense_units'],
                activation=self.config['activation'])

            # what is visualized
            self.label = self.config['dense_units']

        elif self.config['layer_type'] == "Conv1D":
            self.layer = Conv1D(
                filters=self.config['filters'],
                kernel_size=self.config['kernel_size'],
                strides=self.config['strides'],
                padding=self.config['padding'],
                activation=self.config['activation'])

            # what is visualized
            self.label = self.config['filters']

        elif self.config['layer_type'] == "Conv2D":
            self.layer = Conv2D(
                filters=self.config['filters'],
                kernel_size=self.config['kernel_size'],
                strides=self.config['strides'],
                padding=self.config['padding'],
                activation=self.config['activation'])

            # what is visualized
            self.label = self.config['filters']

        elif self.config['layer_type'] == "Dropout":
            self.layer = Dropout(
                rate=self.config['Dropout_rate'])

            # what is visualized
            self.label = self.config['Dropout_rate']
        else:
            raise ValueError("Incorrect layer type")

    def freeze(self):
        self.layer.trainable = False
        self.config['frozen'] = True

    def unfreeze(self):
        self.layer.trainable = True
        self.config['frozen'] = False

    def save(self):
        """
        Save all the configurations and the weights if existent.
        """

        # if the weights can be detected, save them

        weights = self.layer.get_weights()

        wname = self.path + ".weights"

        if exists(wname):
            remove(wname)
        pickle.dump(weights, open(wname, "wb"))

        iname = self.path + ".info"
        if exists(iname):
            remove(iname)
        pickle.dump(self.config, open(iname, "wb"))

    def load(self):

        wname = self.path + ".weights"
        self.weights = pickle.load(open(wname, "rb"))

        iname = self.path + ".info"
        self.config = pickle.load(open(iname, "rb"))

        self.assemble()

    def resetWeights(self):
        raise NotImplemented()
