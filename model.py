import logging

import numpy as np
import tensorflow as tf

import feature
import datasets
import util

logger = logging.getLogger(__name__)


class Model(object):
    def __init__(self):
        self.n_fft = 2048
        self.sr = 11025
        self.win_size = 64
        self.hop_size = self.win_size * 15 // 16

    def network(self, input_shape, output_shape):
        net = tf.reshape(features["X"], [-1, 28, 28, 1])

    def train(self, dataset):
        self.label_dict = dataset.meta['label_dict']
        if isinstance(dataset, str):
            # If dataset is a path to a saved dataset, load it
            dataset = datasets.Dataset(dataset, ['training'], self.prepreprocessor).splits['train']
        if isinstance(dataset , datasets.Dataset):
            dataset = dataset.splits['training']
        dataset = dataset.map(self.preprocessor_py)
        with tf.Session() as sess:
            next_element = dataset.make_one_shot_iterator().get_next()
            while True:
                try:
                    x, y = sess.run(next_element)
                    logger.info(x)
                    logger.info(y)
                    logger.info(x.shape)
                    logger.info(y.shape)
                except:
                    break

    @property
    def prepreprocessor(self):
        def f(x, y=None):
            x = feature.mel_spec(x, n_fft=self.n_fft, sr=self.sr)
            return x, y
        return f

    @property
    def preprocessor(self):
        #TODO make work
        def f(x, y=None):
            x.set_shape([64*10, 128])
            X = feature.split_spec_tf(x, win_size=self.win_size, hop_size=self.hop_size)
            if y is not None:
                print(y.shape)
                y = tf.one_hot(y, 500, dtype=tf.uint8)
                print(y.shape)
                Y = []
                logger.info(X.shape)
                for _ in range(X.shape[0]):
                    Y.append(y)
                Y = tf.stack(Y)
                print(Y)
                print(Y.shape)
            return X, Y
        return f

    @property
    def preprocessor_py(self):
        def f(x, y=None):
            X = feature.split_spec(x, win_size=self.win_size, hop_size=self.hop_size)
            if y is not None:
                y = sum(np.eye(len(self.label_dict))[y])
                Y = []
                for _ in range(len(X)):
                    Y.append(y)
                Y = np.stack(Y).astype(np.uint8)
            X = np.stack(X)
            return X, Y
        return util.wrap_tf_py_func(f, Tout=[tf.float32, tf.uint8])
