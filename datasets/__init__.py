from bidict import bidict
import numpy as np
import tensorflow as tf
import logging
import collections

logger = logging.getLogger(__name__)


class Dataset(object):
    def __init__(self):
        X, Y = self.get_dataset()

    def dump(self, output, split, preprocessor=None):
        X = np.array(self.splits[split]['X'])
        Y = self.splits[split]['Y']
        features_placeholder = tf.placeholder(X.dtype, X.shape)
        labels_placeholder = tf.placeholder(dtype='uint32')
        dataset = tf.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder))
        if preprocessor:
            dataset = dataset.map(preprocessor)
        dataset = dataset.repeat(1)
        iterator = dataset.make_initializable_iterator()
        next_element = iterator.get_next()
        with tf.Session() as sess, tf.python_io.TFRecordWriter(output) as writer:
            feed_dict = {
                features_placeholder: X,
                labels_placeholder: Y,
            }
            sess.run(iterator.initializer, feed_dict=feed_dict)
            while True:
                try:
                    x, y = sess.run(next_element)
                    logger.info(x)
                    logger.info(y)
                    feature = {
                        'X': _bytes_feature(x),
                        'Y': _int64_feature(y),
                    }
                    example = tf.train.Example(features=tf.train.Features(feature=feature))
                    writer.write(example.SerializeToString())
                except tf.errors.OutOfRangeError:
                    break


def indexify(labels, label_dict=None):
    if label_dict is None:
        distinct = set()
        for y in labels:
            if isinstance(y, str):
                distinct.add(y)
            else:
                for label in y:
                    distinct.add(label)
        distinct = sorted(distinct)
        label_dict = bidict((label, index) for index, label in enumerate(distinct))
    logger.info(f'Converting labels to index range: [{min(label_dict.values())}-{max(label_dict.values())}]')
    y = [map_if_collection(label_dict.get, label) for label in labels]
    return y, label_dict


def one_hotify(labels, label_dict=None):
    if label_dict is None:
        labels, label_dict = indexify(labels, label_dict)
    y = np.vstack([one_hot(label, label_dict.inv) for label in labels])
    return y, label_dict


def one_hot(index_labels, label_dict):
    y = np.zeros([1, len(label_dict)], dtype=np.dtype('uint8'))
    y[0, index_labels] = 1
    return y


def one_hot_to_some_hot(Y):
    return Y/Y.sum(axis=1, keepdims=True)


def map_if_collection(func, obj):
    if not isinstance(obj, (str, bytes)) and isinstance(obj, collections.Iterable):
        return tuple(map(func, obj))
    else:
        return func(obj)


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
