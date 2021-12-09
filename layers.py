import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU, Reshape, Softmax
from tensorflow.keras.activations import tanh, sigmoid
from tensorflow.keras.models import Sequential
import diffrend
from generator import THIN_MIN_SIZE


class ConvLReluBn(tf.keras.layers.Layer):
    def __init__(self,
                 filters,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 dilatation=(1, 1),
                 padding="SAME",
                 a=0.1,
                 name=""):
        super().__init__(name=name)
        self.conv = Conv2D(filters, kernel_size=kernel_size, strides=strides, dilation_rate=dilatation, padding=padding)
        self.lrelu = LeakyReLU(a)
        self.bn = BatchNormalization()

    def call(self, x, training=None, **kwargs):
        x = self.conv(x, training=training)
        x = self.lrelu(x, training=training)
        x = self.bn(x, training=training)
        return x


class ResBlock(tf.keras.layers.Layer):
    def __init__(self,
                 n_layers,
                 filters,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 dilatation=(1, 1),
                 padding="SAME",
                 a=0.1,
                 name=""):
        super().__init__(name=name)
        self.layers = Sequential(
            [ConvLReluBn(filters, kernel_size, strides, dilatation, padding, a) for _ in range(n_layers)]
        )

    def call(self, x, training=None, **kwargs):
        y = self.layers(x, training=training)
        y = tf.keras.layers.Concatenate(axis=-1)([y, x])
        return y


class Normalizer(tf.keras.layers.Layer):
    def __init__(self, name=""):
        super().__init__(name=name)
        self.min_width = THIN_MIN_SIZE
        self.min_length = 0.0001

    def call(self, x, **kwargs):
        start_coords = x[..., :2]
        end_coords = x[..., 2:4] + self.min_length
        widths = x[..., 4]

        start_coords = tanh(start_coords) * 1.1
        end_coords = tanh(end_coords) * 1.1
        widths = sigmoid(widths)*2. + self.min_width

        rejoined = tf.concat([start_coords, end_coords, widths[..., tf.newaxis]], axis=-1)
        return rejoined


class Renderer(tf.keras.layers.Layer):
    def __init__(self, img_size, initial_blur, name=""):
        super().__init__(name=name)
        self.img_size = img_size
        self.blur = initial_blur

    def call(self, x, training=None, **kwargs):
        blur = self.blur if training else 0.
        img = diffrend.render_walls(x, self.img_size, blur)
        return img
