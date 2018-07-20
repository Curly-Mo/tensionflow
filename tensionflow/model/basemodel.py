import logging
import functools
import os
import shutil
import pickle  # nosec
import tempfile

import numpy as np
import tensorflow as tf

# from tensorflow.contrib import predictor

from tensionflow import feature
from tensionflow import datasets
from tensionflow import util
from tensionflow import processing
from tensionflow.model import base

logger = logging.getLogger(__name__)

# tf.logging._logger.propagate = False


class BaseModel(base.Model):
    def __init__(self, *args, name='BaseModel', **kwargs):
        self.n_fft = 2048
        self.sr = 11025
        self.win_size = 64
        self.hop_size = self.win_size * 15 // 16
        self.learning_rate = 0.001
        self.name = name
        self.metadata = {}
        self.estimator = None
        super().__init__(*args, **kwargs)

    def network(self, features, output_shape, mode):
        # Add channel dimension
        input_layer = tf.expand_dims(features, -1)
        # input_layer = tf.expand_dims(features['features'], -1)
        height = input_layer.shape[-2]
        logger.info('height: %s', height)
        # feature_columns = [tf.feature_column.numeric_column('features', dtype=tf.float32)]
        # input_layer = tf.feature_column.input_layer(features=features, feature_columns=feature_columns)
        # input_layer = tf.contrib.feature_column.sequence_input_layer(
        #     features=features, feature_columns=feature_columns)
        logger.info('input_layer: %s', input_layer)
        net = tf.layers.conv2d(
            inputs=input_layer,
            filters=48,
            kernel_size=[4, height],
            padding='same',
            activation=tf.nn.relu,
        )
        net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2)
        net = tf.layers.conv2d(
            inputs=input_layer,
            filters=48,
            kernel_size=[4, height],
            padding='same',
            activation=tf.nn.relu,
        )
        net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2)
        net = tf.layers.conv2d(
            inputs=input_layer,
            filters=48,
            kernel_size=[4, height],
            padding='same',
            activation=tf.nn.relu,
        )
        # net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2)
        max_pool = tf.reduce_max(net, [1, 2])
        mean_pool = tf.reduce_mean(net, [1, 2])
        logger.info(max_pool)
        logger.info(mean_pool)
        net = tf.concat([max_pool, mean_pool], -1)
        logger.info(net)
        # shape = net.shape
        # logger.info(f'pool shape: {shape}')
        # flat = tf.reshape(net, [-1, shape[1] * shape[2] * shape[3]])
        # net = tf.reshape(net, [-1, tf.Dimension(32) * shape[2] * shape[3]])
        net = tf.layers.dense(inputs=net, units=2048, activation=tf.nn.relu)
        net = tf.layers.dropout(
            inputs=net, rate=0.8, training=mode == tf.estimator.ModeKeys.TRAIN
        )
        net = tf.layers.dense(inputs=net, units=2048, activation=tf.nn.relu)
        net = tf.layers.dropout(
            inputs=net, rate=0.8, training=mode == tf.estimator.ModeKeys.TRAIN
        )
        logits = tf.layers.dense(inputs=net, units=output_shape)
        return logits

    def estimator_spec(self, logits, labels, mode):
        predictions = {
            'logits': logits,
            'classes': tf.argmax(input=logits, axis=1),
            'probabilities': tf.nn.softmax(logits, name='softmax_tensor'),
        }
        export_outputs = {'features': tf.estimator.export.ClassificationOutput(scores=logits)}
        export_outputs = {
            'features': tf.estimator.export.ClassificationOutput(scores=logits)
        }
        export_outputs = {
            'class': tf.estimator.export.ClassificationOutput(
                classes=tf.as_string(predictions['classes'])
            )
        }

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                mode=mode, predictions=predictions, export_outputs=export_outputs
            )

        logger.info('logits shape: %s', logits.shape)
        logger.info('labels shape: %s', labels.shape)
        loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=labels, logits=logits)
        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            train_op = optimizer.minimize(
                loss=loss, global_step=tf.train.get_global_step()
            )
            return tf.estimator.EstimatorSpec(
                mode=mode, loss=loss, train_op=train_op, export_outputs=export_outputs
            )

        if mode == tf.estimator.ModeKeys.EVAL:
            eval_metric_ops = self.metric_ops(labels, logits)
            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                eval_metric_ops=eval_metric_ops,
                export_outputs=export_outputs,
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
        precisions, prec_ops = tf.metrics.precision_at_thresholds(
            labels, logits, thresholds=thresholds
        )
        recalls, rec_ops = tf.metrics.recall_at_thresholds(
            labels, logits, thresholds=thresholds
        )
        for i, thresh in enumerate(thresholds):
            metric_ops[f'precision@thresh_{thresh}'] = (precisions[i], prec_ops[i])
            metric_ops[f'recall@thresh_{thresh}'] = (recalls[i], rec_ops[i])
            # metric_ops[f'f1_score@thresh_{thresh}'] = (precisions[i] * recalls[i]) / (precision[i] + recall[i])
        return metric_ops

    def output_shape(self, labels):
        if labels is not None:
            self.metadata['output_shape'] = labels.shape[-1]
        return self.metadata['output_shape']

    def model_fn(self):
        def f(features, labels, mode):
            logits = self.network(features, self.output_shape(labels), mode)
            estimator_spec = self.estimator_spec(logits, labels, mode)
            return estimator_spec

        return f

    # @property
    # @functools.lru_cache()
    # def estimator(self):
    #     model = tf.estimator.Estimator(model_fn=self.model_fn())
    #     return model

    def input_fn(
        self, dataset, preprocessors=(), batch_size=5, n_epoch=None, buffer_size=10000
    ):
        def f():
            ds = dataset
            for preprocessor in preprocessors:
                print(preprocessor.func)
                ds = preprocessor.apply(ds)
            ds = ds.shuffle(buffer_size=buffer_size)
            ds = ds.batch(batch_size)
            ds = ds.repeat(n_epoch)
            iterator = ds.make_one_shot_iterator().get_next()
            print(iterator)
            return iterator
            # try:
            #     features, labels = iterator
            #     features = {'features': features}
            #     return features, labels
            # except:
            #     features = {'features': iterator}
            #     return features

        return f

    def train(self, dataset):
        self.metadata = dataset.meta
        training = validation = None
        if isinstance(dataset, str):
            # If dataset is a path to a saved dataset, load it
            dataset = datasets.Dataset(
                dataset, ['training'], functools.partial(self.prepreprocessor)
            ).splits['train']
        if isinstance(dataset, datasets.Dataset):
            training = dataset.splits['training']
            validation = dataset.splits['validation']
        estimator = self.estimator
        # estimator.train(input_fn=self.input_fn(training, self.preprocessor), steps=50)
        # if validation:
        #     estimator.evaluate(input_fn=self.input_fn(validation, self.preprocessor), steps=50)
        train_spec = tf.estimator.TrainSpec(
            input_fn=self.input_fn(training, self.preprocessors)
        )
        eval_spec = tf.estimator.EvalSpec(
            input_fn=self.input_fn(validation, self.preprocessors),
            start_delay_secs=10,
            throttle_secs=150,
        )
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    def predict(self, elements):
        tf.Graph().as_default()
        ds = tf.data.Dataset.from_tensor_slices(elements)
        logger.info(ds)
        estimator = self.estimator
        preprocessors = self.prepreprocessors + self.preprocessors
        return estimator.predict(self.input_fn(ds, preprocessors))

    def save(self, output_dir='saved_models', force=False):
        dst = os.path.join(output_dir, self.name)
        try:
            logger.info('Copying %s, to %s', self.estimator.model_dir, dst)
            if force:
                if os.path.exists(output_dir) and output_dir != dst:
                    shutil.rmtree(output_dir)
            shutil.copytree(self.estimator.model_dir, dst)
        except OSError as _:
            logger.error('Directory already exists: %s. (use force=True)', dst)
        with open(self.metafile(dst), 'wb') as handle:
            pickle.dump(self.metadata, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def export(self, base_dir):
        # feature_spec = {'features': tf.FixedLenSequenceFeature([128], dtype=tf.float32, allow_missing=True)}
        # feature_spec = tf.feature_column.make_parse_example_spec([])
        # feature_spec = {'features': tf.VarLenFeature(dtype=tf.float32)}
        # feature_spec = {'features': tf.placeholder(dtype=tf.float32)}
        # feature_spec = {'features': tf.zeros([10, 128], dtype=tf.float32)}
        # serving_input_receiver_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)

        def serving_input_receiver_fn():
            inputs = {
                'features': tf.placeholder(shape=[None, None, 128], dtype=tf.float32)
            }
            return tf.estimator.export.ServingInputReceiver(inputs, inputs)

        # def serving_input_receiver_fn():
        #     features = tf.placeholder(dtype=tf.float32,
        #                              shape=[None, None, 128],
        #                              name='input_example_tensor')
        #     feature_spec = {'features': features}
        #     return tf.estimator.export.ServingInputReceiver(feature_spec, feature_spec)
        # serving_input_receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(feature_spec)
        self.estimator.export_savedmodel(
            base_dir,
            serving_input_receiver_fn,
            assets_extra=None,
            as_text=False,
            checkpoint_path=None,
        )

    def load(self, saved_path=None):
        model_dir = tempfile.mkdtemp(prefix='tensionflow.')
        if saved_path:
            logger.info('Loading metadata from %s', self.metafile(saved_path))
            with open(self.metafile(saved_path), 'rb') as handle:
                self.metadata = pickle.load(handle)  # nosec
            os.rmdir(model_dir)
            logger.info('Copying %s to %s', saved_path, model_dir)
            shutil.copytree(saved_path, model_dir)
        self.estimator = tf.estimator.Estimator(
            model_fn=self.model_fn(), model_dir=model_dir
        )

    # def import(self path):
    #     self.predict_fn = predictor.from_saved_model(path)
    #     # self.estimator = tf.saved_model.loader.load(self.sess, ['serve'], path)

    @property
    def prepreprocessors(self):
        return [
            processing.PythonPreprocessor(
                functools.partial(self.prepreprocessor),
                output_dtypes=[tf.float32, tf.int32],
                output_shapes=([-1, 128], [-1]),
            )
        ]

    @property
    def preprocessors(self):
        return [
            processing.Preprocessor(functools.partial(self.preprocessor), flatten=True)
        ]

    def prepreprocessor(self, x, y=None):
        x = feature.mel_spec(x, n_fft=self.n_fft, sr=self.sr)
        if y:
            return x, y
        return x

    def preprocessor(self, x, y=None):
        if y is not None:
            y = tf.one_hot(y, len(self.metadata['label_dict']), dtype=tf.uint8)
            y = tf.reduce_sum(y, 0)
        slices = feature.split_spec_tf(
            x, label=y, win_size=self.win_size, hop_size=self.hop_size
        )
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
            f,
            Tout=[
                self.metadata['data_struct'][0]['dtype'],
                self.metadata['data_struct'][1]['dtype'],
            ],
        )

    def metafile(self, directory):
        return os.path.join(directory, 'metadata')
