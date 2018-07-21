import logging

import tensorflow as tf

from tensionflow import util

logger = logging.getLogger(__name__)

# tf.logging._logger.propagate = False


class Preprocessor:
    def __init__(self, func, flatten=False):
        self.func = func
        self.flatten = flatten

    def apply(self, dataset):
        ds = dataset.map(self.func)
        if self.flatten:
            ds = ds.flat_map(lambda *x: tf.data.Dataset.from_tensor_slices(x))
        return ds


class PythonPreprocessor(Preprocessor):
    def __init__(self, func, output_dtypes, output_shapes, *args, **kwargs):
        self.output_dtypes = output_dtypes
        self.output_shapes = output_shapes
        super().__init__(func, *args, **kwargs)

    def apply(self, dataset):
        num_inputs = max(len(dataset.output_shapes), 1)
        ds = dataset.map(util.wrap_tf_py_func(self.func, Tout=self.output_dtypes[:num_inputs]))
        ds = ds.map(lambda *args: tuple(tf.reshape(arg, self.output_shapes[i]) for i, arg in enumerate(args)))
        if self.flatten:
            ds = ds.flat_map(lambda *x: tf.data.Dataset.from_tensor_slices(x))
        return ds
