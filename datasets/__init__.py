import collections
import logging
from functools import partial
import pathlib
import os
import pickle

import numpy as np
import tensorflow as tf

import util

logger = logging.getLogger(__name__)


class Dataset(object):
    def __init__(self, filepath=None, splits=['training'], preprocessor=None, indexify_labels=True):
        self.meta = {}
        self.splits = {}
        if filepath:
            logger.info(f'Loading {splits} datasets from {filepath}')
            with open(os.path.join(filepath, 'metadata'), 'rb') as handle:
                self.meta = pickle.load(handle)
            for split in splits:
                self.splits[split] = self.load(filepath, split)
        else:
            for split in splits:
                X, Y = self.load_features_and_labels(split)
                if indexify_labels:
                    Y = self.indexify(Y, split)
                self.splits[split] = self.generate_dataset(X, Y, preprocessor)

    def load_features_and_labels(self, split):
        """Load or generate data in the form (features, labels)"""
        raise NotImplementedError("Override this method to return (X, Y)")

    def indexify(self, labels, split='training'):
        """Convert string labels to indices in the range (0, num_labels)"""
        if split == 'training':
            Y, self.meta['label_dict'] = util.indexify(labels)
        else:
            Y, _ = util.indexify(Y, self.meta['label_dict'])
        return Y

    def generate_dataset(self, X, Y, preprocessor=None):
        """Generate tensorflow Dataset API from features and labels"""
        if preprocessor is None:
            preprocessor = util.identity_func
        logger.info('Determining processed dtypes...')
        x, y = preprocessor(X[0], Y[0])
        x_dtype = tf.as_dtype(np.array(x).dtype)
        y_dtype = tf.as_dtype(np.array(y).dtype)
        logger.info(f'Preprocess dtypes: {(x_dtype, y_dtype)}')
        def gen(X, Y):
            for x, y in zip(X[0:2], Y[0:2]):
                yield preprocessor(x, y)
        data_gen = partial(gen, X, Y)
        dataset = tf.data.Dataset.from_generator(data_gen, (x_dtype, y_dtype))
        return dataset

    def dump(self, output, splits=None, overwrite=False):
        """Save dataset(s) as tfrecords"""
        pathlib.Path(output).mkdir(parents=True, exist_ok=overwrite)
        filename = os.path.join(output, 'metadata')
        logger.info(f"Saving metadata to {filename}")
        with open(filename, 'wb') as handle:
            pickle.dump(self.meta, handle, protocol=pickle.HIGHEST_PROTOCOL)
        if splits is None:
            splits = [key for key in self.splits]
        for split in splits:
            filename = os.path.join(output, f'{split}.tfrecord')
            logger.info(f"Saving split '{split}' to {filename}")
            dataset = self.splits[split]
            with tf.Session() as sess, tf.python_io.TFRecordWriter(filename) as writer:
                next_element = dataset.make_one_shot_iterator().get_next()
                while True:
                    try:
                        x, y = sess.run(next_element)
                        logger.info(x)
                        logger.info(y)
                        features = util.map_if_collection(util._dtype_feature, x)
                        if not isinstance(features, collections.Iterable):
                            features = [features]
                        feature_list = {
                            'feature_list': {
                                'x': tf.train.FeatureList(feature=features),
                            }
                        }
                        context = {
                            'feature': {
                                'y': util._dtype_feature(y),
                            },
                        }
                        logger.info(feature_list)
                        logger.info(context)
                        example = tf.train.SequenceExample(feature_lists=feature_list, context=context)
                        writer.write(example.SerializeToString())
                    except tf.errors.OutOfRangeError:
                        break

    def load(self, filename, split='training', dtypes={'labels': tf.int64, 'features': tf.float32}):
        """Load a dataset from a tfrecord"""
        def _parse_function(example):
            sequence_features = {
                'x': tf.VarLenFeature(dtype=dtypes['features'])
            }
            context_features = {
                'y': tf.VarLenFeature(dtype=dtypes['labels']),
            }
            context, sequence = tf.parse_single_sequence_example(
                example,
                context_features,
                sequence_features
            )
            features = tf.sparse_tensor_to_dense(sequence['x'],
                                                 default_value=util.default_of_type(dtypes['features']))
            labels = tf.sparse_tensor_to_dense(context['y'],
                                               default_value=util.default_of_type(dtypes['labels']))
            return features, labels
        dataset = tf.data.TFRecordDataset(filename)
        dataset = dataset.map(_parse_function)
        return dataset
