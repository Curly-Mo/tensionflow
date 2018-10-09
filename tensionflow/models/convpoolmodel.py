import logging
import os

import numpy as np
import tensorflow as tf

# from tensorflow.contrib import predictor

from tensionflow import feature
from tensionflow import util
from tensionflow.models import base

logger = logging.getLogger(__name__)

# tf.logging._logger.propagate = False


class ConvPoolModel(base.Model):
    def __init__(self, *args, name='ConvPoolModel', **kwargs):
        self.n_fft = 2048
        self.sr = 11025
        self.win_size = 64
        self.hop_size = self.win_size * 15 // 16
        self.learning_rate = 0.001
        super().__init__(*args, name=name, **kwargs)

    def network(self, features, output_shape, mode):
        # Add channel dimension
        input_layer = tf.expand_dims(features, -1)
        # input_layer = tf.expand_dims(features['features'], -1)
        logger.debug('input_shape: %s', input_layer.shape)
        height = input_layer.shape[-2]
        logger.debug('height: %s', height)
        # feature_columns = [tf.feature_column.numeric_column('features', dtype=tf.float32)]
        # input_layer = tf.feature_column.input_layer(features=features, feature_columns=feature_columns)
        # input_layer = tf.contrib.feature_column.sequence_input_layer(
        #     features=features, feature_columns=feature_columns)
        net = input_layer
        net = tf.layers.conv2d(inputs=net, filters=48, kernel_size=[4, height], padding='same', activation=tf.nn.relu)
        net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2)
        net = tf.layers.conv2d(inputs=net, filters=32, kernel_size=[4, height], padding='same', activation=tf.nn.relu)
        net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2)
        net = tf.layers.conv2d(inputs=net, filters=24, kernel_size=[4, height], padding='same', activation=tf.nn.relu)
        # net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2)
        max_pool = tf.reduce_max(net, [1, 2])
        mean_pool = tf.reduce_mean(net, [1, 2])
        logger.debug(max_pool)
        logger.debug(mean_pool)
        net = tf.concat([max_pool, mean_pool], -1)
        logger.debug(net)
        # shape = net.shape
        # logger.info(f'pool shape: {shape}')
        # flat = tf.reshape(net, [-1, shape[1] * shape[2] * shape[3]])
        # net = tf.reshape(net, [-1, tf.Dimension(32) * shape[2] * shape[3]])
        net = tf.layers.dense(inputs=net, units=1024, activation=tf.nn.relu)
        net = tf.layers.dropout(inputs=net, rate=0.8, training=mode == tf.estimator.ModeKeys.TRAIN)
        # net = tf.layers.dense(inputs=net, units=2048, activation=tf.nn.relu)
        # net = tf.layers.dropout(inputs=net, rate=0.8, training=mode == tf.estimator.ModeKeys.TRAIN)
        logits = tf.layers.dense(inputs=net, units=output_shape, activation=None)
        return logits

    def estimator_spec(self, logits, labels, mode):
        predictions = {
            'logits': logits,
            'classes': tf.argmax(input=logits, axis=1),
            'probabilities': tf.nn.softmax(logits, name='softmax_tensor'),
        }
        export_outputs = {'features': tf.estimator.export.ClassificationOutput(scores=logits)}
        export_outputs = {
            'class': tf.estimator.export.ClassificationOutput(classes=tf.as_string(predictions['classes']))
        }

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, export_outputs=export_outputs)

        logger.debug('logits shape: %s', logits.shape)
        logger.debug('labels shape: %s', labels.shape)
        loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=labels, logits=logits)
        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
            for var in tf.trainable_variables():
                tf.summary.histogram(var.name, var)
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op, export_outputs=export_outputs)

        if mode == tf.estimator.ModeKeys.EVAL:
            eval_metric_ops = self.metric_ops(labels, logits)
            return tf.estimator.EstimatorSpec(
                mode=mode, loss=loss, eval_metric_ops=eval_metric_ops, export_outputs=export_outputs
            )
        return None

    def metric_ops(self, labels, logits):
        labels = tf.cast(labels, tf.int64)
        logits = tf.nn.sigmoid(logits)
        metric_ops = {}
        for k in range(1, 5):
            precision = tf.metrics.precision_at_k(labels, logits, k=k)
            recall = tf.metrics.recall_at_k(labels, logits, k=k)
            metric_ops[f'precision@k_{k}'] = precision
            metric_ops[f'recall@k_{k}'] = recall
            # metric_ops[f'f1_score@k_{k}'] = tf.div(tf.multiply(precision, recall), tf.add(precision,recall))
        thresholds = [x / 10.0 for x in range(1, 10)]
        precisions, prec_ops = tf.metrics.precision_at_thresholds(labels, logits, thresholds=thresholds)
        recalls, rec_ops = tf.metrics.recall_at_thresholds(labels, logits, thresholds=thresholds)
        for i, thresh in enumerate(thresholds):
            metric_ops[f'precision@thresh_{thresh}'] = (precisions[i], prec_ops[i])
            metric_ops[f'recall@thresh_{thresh}'] = (recalls[i], rec_ops[i])
            # metric_ops[f'f1_score@thresh_{thresh}'] = (precisions[i] * recalls[i]) / (precision[i] + recall[i])
        return metric_ops

    def prepreprocessor(self, x, y=None):
        x = feature.mel_spec(x, n_fft=self.n_fft, sr=self.sr)
        if y:
            return x, y
        return x

    def preprocessor(self, x, y=None):
        if y is not None:
            y = tf.one_hot(y, len(self.metadata['label_dict']), dtype=tf.uint8)
            y = tf.reduce_sum(y, 0)
        slices = feature.split_spec_tf(x, label=y, win_size=self.win_size, hop_size=self.hop_size)
        return slices

    @property
    def preprocessor_py(self):
        def f(x, y=None):
            X = feature.split_spec(x, win_size=self.win_size, hop_size=self.hop_size)
            if y is not None:
                y = sum(np.eye(len(self.metadata['label_dict']))[y])
                Y = []
                for _ in range(len(X)):
                    Y.append(y)
                Y = np.stack(Y).astype(np.int32)
            X = np.stack(X)
            return X, Y

        return util.wrap_tf_py_func(
            f, Tout=[self.metadata['data_struct'][0]['dtype'], self.metadata['data_struct'][1]['dtype']]
        )

    def metafile(self, directory):
        return os.path.join(directory, 'metadata')
