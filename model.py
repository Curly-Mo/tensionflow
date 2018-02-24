import tensorflow as tf
import numpy as np
from functools import partial
import feature


class Model(object):
    def __init__(self):
        self.n_fft = 2048
        self.sr = 11025
        self.win_size = 64
        self.hop_size = win_size * 15 // 16

    def network(self, input_shape, output_shape):
        net = tf.reshape(features["X"], [-1, 28, 28, 1])

    def prepreprocess(x, y=None):
        n_fft = 2048
        sr = 11025
        x = feature.mel_spec(x.eval(), n_fft=n_fft, sr=sr)
        if y:
            return x, y
        return x

    def preprocess(self, x, y=None):
        pass
