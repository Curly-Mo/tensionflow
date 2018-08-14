import collections
import logging
import tensorflow as tf
import numpy as np
from bidict import bidict

logger = logging.getLogger(__name__)


def identity_func(*args):
    return args


def default_of_type(dtype=int):
    if dtype == tf.string:
        return 'default_value'
    try:
        return dtype()
    except TypeError:
        pass
    try:
        return dtype.as_numpy_dtype()
    except (AttributeError, TypeError):
        pass
    return 0


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
    logger.info('Converting labels to index range: [%s-%s]', min(label_dict.values()), max(label_dict.values()))
    y = [map_if_collection(label_dict.get, label) for label in labels]
    return y, label_dict


def one_hotify(labels, label_dict=None):
    if label_dict is None:
        labels, label_dict = indexify(labels, label_dict)
    y = np.stack([one_hot(label, len(label_dict.inv)) for label in labels])
    return y, label_dict


def one_hot(index_labels, num_labels):
    y = np.zeros([num_labels], dtype=np.dtype('uint8'))
    y[np.asarray(index_labels)] = 1
    return y


def one_hot_to_some_hot(Y):
    return Y / Y.sum(axis=1, keepdims=True)


def map_if_collection(func, obj):
    if not isinstance(obj, (str, bytes)) and isinstance(obj, collections.Iterable):
        return tuple(map(func, obj))
    else:
        return func(obj)


def _dtype_feature(ndarray):
    """match appropriate tf.train.Feature class with dtype of ndarray. """
    if not isinstance(ndarray, np.ndarray):
        logger.info('Converting %s to ndarray', ndarray)
        ndarray = np.array([ndarray])
    dtype = ndarray.dtype
    if np.issubdtype(dtype, np.float):
        return tf.train.Feature(float_list=tf.train.FloatList(value=ndarray))
    elif np.issubdtype(dtype, np.integer):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=ndarray))
    else:
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=ndarray))


def wrap_tf_py_func(py_func, Tout):
    def f(*args):
        return tf.py_func(py_func, inp=args, Tout=Tout)

    return f


def get_all_subclasses(cls):
    all_subclasses = {}
    for subclass in cls.__subclasses__():
        all_subclasses[subclass.__name__] = subclass
        all_subclasses.update(get_all_subclasses(subclass))
    return all_subclasses
