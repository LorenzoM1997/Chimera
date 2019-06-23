import tensorflow as tf
import os
import pickle

from tensorflow.keras.layers import Conv2D, Dense, Flatten, Reshape
from tensorflow.keras import Model

from Layers import Strato


class Chimera(object):

    def __init__(self):

        self.loss = tf.keras.losses.CategoricalCrossentropy()
        self.optimizer = tf.keras.optimizers.Adam()

        self.inputShape = None
        self.outputShape = None

        self.layers = []
        self.filename = ""

        # create the directories if they are missing
        if not os.path.isdir("models"):
            os.mkdir("models")

    def build(self):
        # define input layer
        inputs = tf.keras.Input(shape=(self.inputShape,))

        # define all other layers
        if self.layers[0].layer_type == 'Conv1D':
            x = Reshape(target_shape=(1, self.inputShape))(inputs)
            x = self.layers[0].layer(x)
        else:
            x = self.layers[0].layer(inputs)

        for i in range(1, len(self.layers)):
            if self.layers[i].layer_type == "Dense" and self.layers[i-1].layer_type != "Dense":
                x = Flatten()(x)
            x = self.layers[i].layer(x)

        # define output layer

        outputs = tf.keras.layers.Dense(
            self.outputShape, activation=tf.nn.softmax)(x)

        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)

        # extract default weights
        self.weights = self.model.get_weights()

        # if the weights can be loaded, load them
        for i in range(len(self.layers)):
            if self.layers[i].weights != None:
                self.weights.append(self.layers[i].weights)
            else:
                print("weights not found")

        # set loaded weigts
        self.model.set_weights(self.weights)

        self.model.compile(
            optimizer=self.optimizer,
            loss=self.loss,
            metrics=['accuracy'])

    def add_layer(self, layer_type, ix=None):

        if ix == None:
            ix = len(self.layers)

        self.layers.insert(ix, Strato(layer_type=layer_type))

    """
    move the layer one position up
    args:
        ix (int): the index of the layer we are moving up
    """

    def move_up(self, ix):
        if ix <= 0 or ix >= len(self.layers):
            raise ValueError("Layer index out of bound")
        else:
            l = self.layers.pop(ix)
            self.layers.insert(ix - 1, l)

    def move_down(self, ix):
        if ix < 0 or ix >= len(self.layers) - 1:
            raise ValueError("Layer index out of bound")
        else:
            l = self.layers.pop(ix)
            self.layers.insert(ix + 1, l)

    def remove_layer(self, ix):
        if ix < 0 or ix >= len(self.layers):
            raise ValueError("Layer index out of bound")
        else:
            self.layers.pop(ix)

    def defineInputShape(self, x):
        self.inputShape = x.shape[1]

    def defineOutputShape(self, y):
        self.outputShape = y.shape[1]

    def fit(self, x, y, batch_size=1, epochs=10):
        """
        function to train the model given the input and labels
        Args:
            x : the input
            y : the labels
            batch_size (int)
            epochs (int) : the number of times we iterate through the entire data set
        """

        # make sure the number of samples is the same
        assert x.shape[0] == y.shape[0]

        # automatically detect input shape
        if self.inputShape == None:
            self.defineInputShape(x)

        if self.outputShape == None:
            self.defineOutputShape(y)

        # build the model
        self.build()

        # fit to the data
        history_obj = self.model.fit(
            x=x,
            y=y,
            batch_size=batch_size,
            epochs=epochs,
            verbose=0   # we don't want any output to screen
        )

        return history_obj

    def predict(self, x):
        """
        predicts the output using the current model
        Args:
            x: the input
        """
        self.build()

        return self.model.predict(x)

    def load(self, name):
        """
        load the model if it exists
        Args:
            name (str): the name of the model
        """

        if name == "":
            return

        modelpath = os.path.join("models", name)
        if os.path.exists(modelpath):
            layerList = pickle.load(open(modelpath, "rb"))
            self.layers = []
            for l in layerList:
                s = Strato(name=l)
                self.layers.append(s)

    def save(self, name=""):
        """
        save the model
        Args:
            name (str) the name to use for the model
        """

        if name != "":
            self.filename = name

        if self.filename == "":
            raise ValueError("Model name not defined")

        # the names of the layers are appended to a list
        layerList = []
        for l in self.layers:
            l.save()
            layerList.append(l.name)

        modelpath = os.path.join("models", self.filename)
        pickle.dump(layerList, open(modelpath, "wb"))
