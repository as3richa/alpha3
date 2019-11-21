from tensorflow import keras
from tensorflow.keras import layers, regularizers, initializers

initializer = initializers.glorot_uniform(seed=None)

class ConvolutionalBlock(layers.Layer):
    def __init__(self, filters, kernel_size, l2_reg):
        super(ConvolutionalBlock, self).__init__()

        reg = regularizers.l2(l2_reg)

        self._convolution = layers.Conv2D(filters, kernel_size, data_format="channels_last", padding="same", kernel_regularizer=reg)
        self._batch_norm = layers.BatchNormalization()
        self._activation = layers.Activation("relu")

    def call(self, inputs):
        return self._activation(self._batch_norm(self._convolution(inputs)))

class ResidualBlock(layers.Layer):
    def __init__(self, filters, kernel_size, l2_reg):
        super(ResidualBlock, self).__init__()

        reg = regularizers.l2(l2_reg)

        self._first_conv, self._second_conv = (ConvolutionalBlock(filters, kernel_size, l2_reg) for _ in range(2))
        self._add = layers.Add()

    def call(self, inputs):
        return self._add([inputs, self._second_conv(self._first_conv(inputs))])

class ConvNet3x3(keras.Model):
    def __init__(self, columns, l2_reg=0.01):
        super(ConvNet3x3, self).__init__()

        filters = 64
        kernel_size = 3

        self._tower = [ConvolutionalBlock(filters, kernel_size, l2_reg)]
        self._tower.extend(ResidualBlock(filters, kernel_size, l2_reg) for _ in range(4))

        reg = regularizers.l2(l2_reg)

        self._outcome_head = [
            ConvolutionalBlock(1, 1, l2_reg),
            layers.Flatten(data_format="channels_last"),
            layers.Dense(64, activation="relu", kernel_regularizer=reg, bias_initializer=initializer),
            layers.Dense(1, activation="tanh", kernel_regularizer=reg, bias_initializer=initializer)
        ]

        self._policy_head = [
            ConvolutionalBlock(2, 1, l2_reg),
            layers.Flatten(data_format="channels_last"),
            layers.Dense(columns, activation="softmax", kernel_regularizer=reg, bias_initializer=initializer)
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
