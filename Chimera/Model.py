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
        self.changeList = []
        self.filename = ""

        # create the directories if they are missing
        if not os.path.isdir("models"):
            os.mkdir("models")

    def build(self):

        # check that all layers have correct shape
        self.checkLayerShape()

        # define input layer
        inputs = tf.keras.Input(shape=(self.inputShape,))

        x = tf.identity(inputs)

        # define all other layers
        for i in range(len(self.layers)):

            thisLayerType = self.layers[i].config['layer_type']

            if i > 0:
                lastLayer = self.layers[i - 1].config

                if thisLayerType == "Dense" and lastLayer['layer_type'] != "Dense":
                    x = Flatten()(x)
                elif thisLayerType == "Conv1D" and lastLayer['layer_type'] == "Dense":
                    x = Reshape(target_shape=(1, lastLayer['dense_units']))(x)
            else:
                # if it's the first layer and it's convolutional, reshape
                if thisLayerType == "Conv1D":
                    x = Reshape(target_shape=(1, self.inputShape))(x)

            x = self.layers[i].layer(x)

        self.model = tf.keras.Model(inputs=inputs, outputs=x)

        # extract default weights
        self.weights = self.model.get_weights()

        # if the weights can be loaded, load them
        for i in range(len(self.layers)):
            if self.layers[i].weights is not None:
                self.weights.append(self.layers[i].weights)
            else:
                print("weights not found")

        # set loaded weigts
        self.model.set_weights(self.weights)

        self.model.compile(
            optimizer=self.optimizer,
            loss=self.loss,
            metrics=['accuracy'])

    def checkLayerShape(self):

        for i in range(len(self.changeList)):
            if self.changeList[i] == True:

                # if there is a change, reassemble the layer
                self.layers[i].assemble()
                self.changeList[i] = False
            
    def add_layer(self, layer_type, ix=None):

        config = {'layer_type': layer_type}

        if ix is None:
            if len(self.layers) == 0:
                ix = len(self.layers)
                config['layer_type'] = 'Dense'
                if self.outputShape is not None:
                    config['dense_units'] = self.outputShape
            else:
                ix = len(self.layers) - 1

        self.layers.insert(ix, Strato(config=config))

        # update the list of changes
        self.changeList.insert(ix, False)
        if ix + 1 < len(self.changeList):
            self.changeList[ix + 1] = True

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

            # mark the two layers as changed
            self.changeList[ix] = True
            self.changeList[ix - 1] = True
            if ix + 1 < len(self.changeList):
                self.changeList[ix + 1] = True

    def move_down(self, ix):
        if ix < 0 or ix >= len(self.layers) - 1:
            raise ValueError("Layer index out of bound")
        else:
            l = self.layers.pop(ix)
            self.layers.insert(ix + 1, l)

            # mark the two layers as changed
            self.changeList[ix + 1] = True
            self.changeList[ix] = True
            if ix + 2 < len(self.changeList):
                self.changeList[ix + 2] = True

    def remove_layer(self, ix):
        if ix < 0 or ix >= len(self.layers):
            raise ValueError("Layer index out of bound")
        else:
            # pop the layer out
            self.layers.pop(ix)

            # mark the following layer as changed
            self.changeList.pop(ix)
            self.changeList[ix] = True

    def defineOutputShape(self, outputShape):
        self.outputShape = outputShape

        # enforce the last layer shape to match the output shape
        lastLayer = self.layers[-1]
        lastLayer.config['layer_type'] = "Dense"
        lastLayer.config['dense_units'] = self.outputShape
        lastLayer.assemble()

    def fit(self, x, y, batch_size=1, epochs=10):
        """
        function to train the model given the input and labels
        Args:
            x : the input
            y : the labels
            batch_size (int)
            epochs (int) : the number of times we iterate through the entire data set
        """

        # automatically detect input shape
        if self.inputShape is None or self.inputShape != inputShape:
            self.inputShape = x.shape[0]

        if self.outputShape is None or self.outputShape != outputShape:
            self.defineOutputShape(y.shape[0])

        # build the model
        self.build()

        # fit to the data
        history_obj = self.model.fit(
            train_dataset,
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

            # reset all the layer information
            self.layers = []
            self.changeList = []

            # append each layer
            for l in layerList:
                s = Strato(name=l)
                self.layers.append(s)
                self.changeList.append(False)

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

    def export(self, filepath):

        # rebuild the model to be sure it's up to date
        try:
            self.build()
        except:
            raise RuntimeError()
            return

        # save the model in h5 format
        self.model.save(
            filepath,
            overwrite=True,
            include_optimizer=True,
            save_format = "h5"
        )