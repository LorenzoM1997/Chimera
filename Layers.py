import os
import random
import string
import tensorflow as tf
import numpy as np
import pickle

from tensorflow.keras.layers import Layer, Dense, Flatten, Conv2D

def check_dir():
    if not os.path.isdir("layers"):
        os.mkdir("layers")

class Strato(object):
    def __init__(self, name=None, layer_type="Dense"):
        self.layer_type = layer_type

        self.assemble()

        self.frozen = False

        check_dir()
        
        if name == None:
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
                units=4,
                activation='relu')

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
        weights = pickle.load(open(wname, "rb"))

        iname = filename + ".info"
        info = pickle.load(open(iname, "rb"))
        self.layer_type = info['layer_type']
        self.frozen = info['frozen']

        self.assemble()
        self.layer.build(info['input_shape'])
        self.layer.set_weights(weights)

    def resetWeights(self):
        raise NotImplemented()
