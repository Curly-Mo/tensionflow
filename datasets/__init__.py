import collections
import logging
from functools import partial

import numpy as np
import tensorflow as tf

import util

logger = logging.getLogger(__name__)


class Dataset(object):
    def __init__(self, filepath=None):
        if filepath:
            self.load(filepath)
        pass

    def dump(self, output, split=None, preprocessor=util.identity_func):
        if split:
            splits = [split]
        else:
            splits = [key for key in self.splits]
        for split in splits:
            logger.info(f'Dumping split: {split}')
            X = self.splits[split]['X']
            Y = self.splits[split]['Y']
            logger.info('Determining processed dtypes...')
            x, y = preprocessor(X[0], Y[0])
            x_dtype = tf.as_dtype(np.array(x).dtype)
            y_dtype = tf.as_dtype(np.array(y).dtype)
            logger.info(f'Preprocess dtypes: {(x_dtype, y_dtype)}')
            def gen(X, Y):
                for x, y in zip(X[0:2], Y[0:2]):
                    yield preprocessor(x, y)
            data_gen = partial(gen, X, Y)
            with tf.Session().as_default() as sess, tf.python_io.TFRecordWriter(output) as writer:
                dataset = tf.data.Dataset.from_generator(data_gen, (x_dtype, y_dtype))
                next_element = dataset.make_one_shot_iterator().get_next()
                while True:
                    try:
                        x, y = sess.run(next_element)
                        logger.info(x)
                        logger.info(y)
                        features = util.map_if_collection(util._dtype_feature, x)
                        if not isinstance(features, collections.Iterable):
                            features = [features]
                        logger.info(features)
                        feature_list = {
                            'feature_list': {
                                'features': tf.train.FeatureList(feature=features),
                            }
                        }
                        logger.info(feature_list)
                        context = {
                            'feature': {
                                'labels': util._dtype_feature(y),
                            },
                        }
                        example = tf.train.SequenceExample(feature_lists=feature_list, context=context)
                        writer.write(example.SerializeToString())
                    except tf.errors.OutOfRangeError:
                        break

    def load(self, filename, dtypes={'labels': tf.int64, 'features': tf.float32}):
        def _parse_function(example):
            context_features = {
                'labels': tf.VarLenFeature(dtype=dtypes['labels']),
            }
            sequence_features = {
                'features': tf.VarLenFeature(dtype=dtypes['features'])
            }
            context, sequence = tf.parse_single_sequence_example(
                example,
                context_features,
                sequence_features
            )
            features = tf.sparse_tensor_to_dense(sequence['features'],
                                                 default_value=util.default_of_type(dtypes['features']))
            labels = tf.sparse_tensor_to_dense(context['labels'],
                                               default_value=util.default_of_type(dtypes['labels']))
            return features, labels
        dataset = tf.data.TFRecordDataset(filename)
        dataset = dataset.map(_parse_function)
        return dataset
