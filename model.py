import logging

import numpy as np
import tensorflow as tf

import feature
import datasets
import util

logger = logging.getLogger(__name__)
tf.logging._logger.propagate = False


class Model(object):

    def __init__(self):
        self.n_fft = 2048
        self.sr = 11025
        self.win_size = 64
        self.hop_size = self.win_size * 15 // 16

    def network(self, input_layer, output_shape, mode):
        # Add channel dimension
        input_layer = tf.expand_dims(input_layer, -1)
        net = tf.layers.conv2d(
            inputs=input_layer, filters=48, kernel_size=[4, 128], padding='same', activation=tf.nn.relu)
        net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2)
        shape = net.shape
        logger.info(f'pool shape: {shape}')
        flat = tf.reshape(net, [-1, shape[1] * shape[2] * shape[3]])
        net = tf.layers.dense(inputs=flat, units=1024, activation=tf.nn.relu)
        net = tf.layers.dropout(inputs=net, rate=0.6, training=mode == tf.estimator.ModeKeys.TRAIN)
        logits = tf.layers.dense(inputs=net, units=114)
        return logits

    def estimator_spec(self, logits, labels, mode):
        logger.info(f'logits shape: {logits.shape}')
        logger.info(f'labels shape: {labels.shape}')
        predictions = {
            'classes': tf.argmax(input=logits, axis=1),
            'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
        }
        logger.info(f'predictions: {predictions}')
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
        loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=labels, logits=logits)
        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
            train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

        if mode == tf.estimator.ModeKeys.EVAL:
            labels = tf.cast(labels, tf.int64)
            eval_metric_ops = {'accuracy': tf.metrics.accuracy(labels=labels, predictions=predictions['probabilities'])}
            for k in range(1, 6):
                precision = tf.metrics.precision_at_k(labels, logits, k=k)
                recall = tf.metrics.precision_at_k(labels, logits, k=k)
                eval_metric_ops[f'precision@{k}'] = precision
                eval_metric_ops[f'recall@{k}'] = recall
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

    def model_fn(self):

        def f(features, labels, mode):
            logits = self.network(features, labels.shape[-1], mode)
            estimator_spec = self.estimator_spec(logits, labels, mode)
            return estimator_spec

        return f

    def estimator(self):
        model = tf.estimator.Estimator(model_fn=self.model_fn(), model_dir='/tmp/test_model')
        return model

    def input_fn(self, dataset):

        def f():
            ds = dataset.map(self.preprocessor)
            ds = ds.flat_map(lambda *x: tf.data.Dataset.from_tensor_slices(x))
            ds = ds.shuffle(buffer_size=10000)
            ds = ds.batch(5)
            ds = ds.repeat(1)
            features, labels = ds.make_one_shot_iterator().get_next()
            logger.info(f'features shape: {features.shape}')
            logger.info(f'labels shape: {labels.shape}')
            return features, labels

        return f

    def train(self, dataset):
        self.metadata = dataset.meta
        training = validation = None
        if isinstance(dataset, str):
            # If dataset is a path to a saved dataset, load it
            dataset = datasets.Dataset(dataset, ['training'], self.prepreprocessor).splits['train']
        if isinstance(dataset, datasets.Dataset):
            training = dataset.splits['training']
            validation = dataset.splits['validation']
        estimator = self.estimator()
        estimator.train(input_fn=self.input_fn(training), steps=50)
        if validation:
            estimator.evaluate(input_fn=self.input_fn(validation), steps=50)

    @property
    def prepreprocessor(self):

        def f(x, y=None):
            x = feature.mel_spec(x, n_fft=self.n_fft, sr=self.sr)
            return x, y

        return f

    @property
    def preprocessor(self):

        def f(x, y=None):
            X = feature.split_spec_tf(
                x,
                win_size=self.win_size,
                hop_size=self.hop_size,
                max_len=self.metadata['data_struct'][0]['min_shape'][0])
            logger.info(X.shape)
            if y is not None:
                y = tf.one_hot(y, len(self.metadata['label_dict']), dtype=tf.uint8)
                logger.info(y.shape)
                y = tf.reduce_sum(y, 0)
                logger.info(y.shape)
                Y = []
                for _ in range(X.shape[0]):
                    Y.append(y)
                Y = tf.stack(Y)
            logger.info(Y.shape)
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
                Y = np.stack(Y).astype(np.int32)
            X = np.stack(X)
            return X, Y

        return util.wrap_tf_py_func(f, Tout=[tf.float32, tf.int32])
