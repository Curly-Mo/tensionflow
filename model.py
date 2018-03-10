import tensorflow as tf
import numpy as np
from functools import partial
import feature


class Model(object):
    def __init__(self):
        self.n_fft = 2048
        self.sr = 11025
        self.win_size = 64
        self.hop_size = self.win_size * 15 // 16

    def network(self, input_shape, output_shape):
        net = tf.reshape(features["X"], [-1, 28, 28, 1])

    @property
    def prepreprocessor(self):
        def f(x, y=None):
            x = feature.mel_spec(x, n_fft=self.n_fft, sr=self.sr)
            return x, y
        return f

    @property
    def preprocessor(self):
        def f(x, y=None):
            if y:
                y = tf.one_hot(y)
            n_fft = 2048
            sr = 11025
            #x = feature.mel_spec(x, n_fft=n_fft, sr=sr)
            return [1,2,3], y
        return tf.py_func(f, inp=(x, y), Tout=[tf.float32, tf.uint8])
