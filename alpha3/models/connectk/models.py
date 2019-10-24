from tensorflow import keras
from tensorflow.keras import layers

class ConvolutionalBlock(layers.Layer):
    def __init__(self, filters, kernel_size):
        super(ConvolutionalBlock, self).__init__()
        self._convolution = layers.Conv2D(filters, kernel_size, data_format="channels_last", padding="same")
        self._batch_norm = layers.BatchNormalization()
        self._activation = layers.Activation("relu")

    def call(self, inputs):
        return self._activation(self._batch_norm(self._convolution(inputs)))

class ResidualBlock(layers.Layer):
    def __init__(self, filters, kernel_size):
        super(ResidualBlock, self).__init__()
        self._first_conv, self._second_conv = (ConvolutionalBlock(filters, kernel_size) for _ in range(2))
        self._add = layers.Add()

    def call(self, inputs):
        return self._add([inputs, self._second_conv(self._first_conv(inputs))])

class ConvNet3x3(keras.Model):
    def __init__(self, rows, columns):
        super(ConvNet3x3, self).__init__()

        filters = 64
        kernel_size = 3

        self._tower = [ConvolutionalBlock(filters, kernel_size)]
        self._tower.extend(ResidualBlock(filters, kernel_size) for _ in range(4))

        self._outcome_head = [
            ConvolutionalBlock(1, 1),
            layers.Flatten(data_format="channels_last"),
            layers.Dense(64, activation="relu"),
            layers.Dense(1, activation="tanh")
        ]

        self._policy_head = [
            ConvolutionalBlock(2, 1),
            layers.Flatten(data_format="channels_last"),
            layers.Dense(columns, activation="softmax")
        ]

        self._concat = layers.Concatenate()

    def call(self, inputs):
        features = inputs

        for layer in self._tower:
            features = layer(features)

        predicted_outcome = features

        for layer in self._outcome_head:
            predicted_outcome = layer(predicted_outcome)

        policy = features

        for layer in self._policy_head:
            policy = layer(policy)

        return self._concat([predicted_outcome, policy])
