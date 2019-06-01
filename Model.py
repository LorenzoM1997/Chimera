import tensorflow as tf

from tensorflow.keras.layers import Conv2D, Dense, Flatten
from tensorflow.keras import Model

from Layers import Strato


class Chimera(object):
    def __init__(self):

        self.loss = tf.keras.losses.CategoricalCrossentropy()
        self.optimizer = tf.keras.optimizers.Adam()

        self.inputShape = None
        self.outputShape = None

        self.layers = []

    def build(self):
        # define input layer
        inputs = tf.keras.Input(shape=(self.inputShape,))

        # define all other layers
        x = self.layers[0].layer(inputs)

        for i in range(1, len(self.layers)):
            x = self.layers[i].layer(x)

        # define output layer
        outputs = tf.keras.layers.Dense(
            self.outputShape, activation=tf.nn.softmax)(x)

        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)

        self.model.compile(
            optimizer=self.optimizer,
            loss=self.loss,
            metrics=['accuracy'])

        self.model.summary()

    def add_layer(self, layer_type, ix=None):

        if ix == None:
            ix = len(self.layers)

        self.layers.insert(ix, Strato(layer_type=layer_type))

    def defineInputShape(self, x):
        self.inputShape = x.shape[1]

    def defineOutputShape(self, y):
        self.outputShape = y.shape[1]

    def fit(self, x, y, batch_size=1, epochs=10, verbose=1):

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
        self.model.fit(
            x=x,
            y=y,
            batch_size=batch_size,
            epochs=epochs,
            verbose = verbose)

    def predict(self, x):
        return self.model.predict(x)
