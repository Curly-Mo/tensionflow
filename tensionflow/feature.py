import logging
import collections
import functools

import numpy as np
import librosa
import tensorflow as tf

logger = logging.getLogger(__name__)


def one_hot(label_ids, label_dict):
    indices = [label_dict[id] for id in label_ids]
    y = np.zeros([1, len(label_dict)])
    y[0, indices] = 1
    return y


def some_hot(label_ids, label_dict):
    indices = [label_dict[id] for id in label_ids]
    y = np.zeros([1, len(label_dict)])
    y[0, indices] = 1 / len(label_ids)
    return y


def one_hot_to_some_hot(Y):
    return Y / Y.sum(axis=1, keepdims=True)


def distinct_from_labels(Y):
    distinct = set()
    for y in Y:
        if isinstance(y, str):
            distinct.add(y)
        else:
            for label in y:
                distinct.add(label)
    labels = sorted(distinct)
    label_dict = {label: index for index, label in enumerate(labels)}
    return labels, label_dict


def flatmap(func, iterable):
    for item in iterable:
        try:
            yield func(item)
        except:  # noqa pylint: disable=bare-except
            print(f'Unable to {func} {item}')


def training_features(x_files, y_labels):
    labels, label_dict = distinct_from_labels(y_labels)
    win_size = 64
    hop_size = win_size * 15 // 16
    y_train = np.vstack([one_hot(y, label_dict) for y in y_labels])

    x_train_spec = flatmap(mel_spec, x_files)
    X = []
    Y = []
    for i, (x, y) in enumerate(zip(x_train_spec, y_train)):
        print(i)
        x = split_spec(x, win_size, hop_size)
        if x is not None:
            X.append(x)
            for _ in range(len(x)):
                Y.append(y)
    X = np.vstack(X)
    Y = np.vstack(Y)
    X = X[..., np.newaxis]
    return X, Y, labels


def mel_spec(audio_path, n_fft=2048, sr=11025):
    y, sr = librosa.load(audio_path, mono=True, sr=sr)
    y, _ = librosa.effects.trim(y)
    melspec = librosa.feature.melspectrogram(y, n_fft=n_fft)
    melspec = np.float32(melspec)
    return melspec.T


def split_spec(S, win_size, hop_size):
    X = []
    i = 0
    while i + win_size < len(S):
        x = S[i : i + win_size]
        X.append(x)
        i += hop_size
    if not X:
        return np.empty(S.shape)
    return np.stack(X)


def split_spec_tf(S, win_size, hop_size, label=None):
    length = tf.shape(S)[0]
    logger.debug(length)
    X = tf.TensorArray(dtype=S.dtype, infer_shape=False, size=0, dynamic_size=True)

    def cond(i, index, X, Y=None):  # pylint: disable=unused-argument
        return tf.less(index + win_size, length)

    def body(i, index, X, Y=None):
        temp = S[index : index + win_size, ...]
        X = X.write(i, temp)
        if label is not None:
            Y = Y.write(i, label)
            return i + 1, index + hop_size, X, Y
        return i + 1, index + hop_size, X

    def empty_tensors(X, Y):
        x = tf.constant(0, shape=[0, win_size] + S.shape[1:].as_list(), dtype=X.dtype)
        if Y is not None:
            y = tf.constant(0, shape=[0] + label.shape[min(1, len(label.shape) - 1) :].as_list(), dtype=Y.dtype)
            return x, y
        return x

    Y = None
    if label is not None:
        Y = tf.TensorArray(dtype=label.dtype, infer_shape=False, size=0, dynamic_size=True)
    count, _, X, Y = tf.while_loop(cond, body, [0, 0, X, Y], parallel_iterations=1)
    # tensorflow can't stack an empty array, this hack returns an empty tensor of same shape if array is empty
    X, Y = tf.cond(count > 0, lambda: (X.stack(), Y.stack()), functools.partial(empty_tensors, X, Y))
    # reshape so that the shape is known
    X = tf.reshape(X, [-1, win_size] + S.shape[1:].as_list())
    if Y is not None:
        Y = tf.reshape(Y, [-1] + label.shape[min(1, len(label.shape) - 1) :].as_list())
    return X, Y


def balanced_sample(X_in, Y_in):
    X, Y = [], []
    max_count = sum(Y)[np.nonzero(sum(Y))].min()
    counts = collections.defaultdict(int)
    for x, y in zip(X_in, Y_in):
        for indices in y.nonzero():
            if all(counts[i] < max_count for i in indices):
                X.append(x)
                Y.append(y)
                for i in indices:
                    counts[i] += 1
    return np.stack(X), np.stack(Y)
