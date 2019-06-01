import os
import random
import string
import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Layer, Dense, Flatten, Conv2D

def check_dir():
    if not os.path.isdir("layers"):
        os.mkdir("layers")

class Strato(object):
    def __init__(self, name=None, layer_type="Dense"):
        self.layer_type = layer_type

        if layer_type == "Dense":
            self.layer = Dense(
                units=64,
                activation='relu')

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

    def freeze(self):
        self.layer.trainable = False

    def unfreeze(self):
        self.layer.trainable = True

    def save(self):
        weights = self.layer.get_weights()
        check_dir()
        filename = os.path.join("layers", self.name)

        np.save(filename, weights)

    def load(self):
        check_dir()
        filename = os.path.join("layers", self.name)
        filename += ".npy"
        weights = np.load(filename)

        self.layer.set_weights(weights)

    def resetWeights(self):
        raise NotImplemented()
