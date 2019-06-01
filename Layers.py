import os
import random
import string
import tensorflow as tf

from tensorflow.keras.layers import Layer, Dense, Flatten, Conv2D


class Strato(object):
    def __init__(self, name=None, layer_type="Dense"):
        self.layer_type = layer_type

        if layer_type == "Dense":
            self.layer = Dense(
                units=64,
                activation='relu')

        self.frozen = False

        if not os.path.isdir("layers"):
            os.mkdir("layers")

        if name == None:
            self.name = ''.join(random.choice(
                string.ascii_lowercase + string.digits) for _ in range(16))

            self.path = os.path.join("layers", self.name)
            while os.path.exists(self.path):
                self.name = ''.join(random.choice(
                    string.ascii_lowercase + string.digits) for _ in range(16))
                self.path = os.path.join("layers", self.name)
        else:
            self.load(name)
            self.name = name

    def freeze(self):
        self.layer.trainable = True

    def unfreeze(self):
        self.layer.trainable = False

    def save(self):
        pass

    def load(self, name):
        raise NotImplemented()

    def resetWeights(self):
        raise NotImplemented()
