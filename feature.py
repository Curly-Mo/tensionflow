import collections
import numpy as np
import librosa


def one_hot(label_ids, label_dict):
    indices = [label_dict[id] for id in label_ids]
    y = np.zeros([1, len(label_dict)])
    y[0, indices] = 1
    return y


def some_hot(label_ids, label_dict):
    indices = [label_dict[id] for id in label_ids]
    y = np.zeros([1, len(label_dict)])
    y[0, indices] = 1/len(label_ids)
    return y


def one_hot_to_some_hot(Y):
    return Y/Y.sum(axis=1, keepdims=True)


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


def flatmap(func,  iterable):
    for item in iterable:
        try:
            yield func(item)
        except:
            print(f'Unable to {func} {item}')
            pass


def training_features(x_files, y_labels):
    labels, label_dict = distinct_from_labels(y_labels)
    win_size = 64
    hop_size = win_size*15//16
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
    print(audio_path)
    y, sr = librosa.load(audio_path, mono=True, sr=sr)
    y, index = librosa.effects.trim(y)
    melspec = librosa.feature.melspectrogram(y, n_fft=n_fft)
    melspec = np.float32(melspec)
    print(melspec.T.shape)
    return melspec.T


def split_spec(S, win_size, hop_size):
    X = []
    i = 0
    while i + win_size < len(S):
        x = S[i:i+win_size]
        X.append(x)
        i += hop_size
    if not X:
        return None
    return np.stack(X)


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
