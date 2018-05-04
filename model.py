import logging
import functools

import numpy as np
import tensorflow as tf

import feature
import datasets
import util
import processing

logger = logging.getLogger(__name__)
tf.logging._logger.propagate = False


class Model(object):

    def __init__(self, name='BaseModel'):
        self.n_fft = 2048
        self.sr = 11025
        self.win_size = 64
        self.hop_size = self.win_size * 15 // 16
        self.name = name

    def network(self, input_layer, output_shape, mode):
        # Add channel dimension
        input_layer = tf.expand_dims(input_layer, -1)
        net = tf.layers.conv2d(
            inputs=input_layer, filters=48, kernel_size=[4, 128], padding='same', activation=tf.nn.relu)
        net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2)
        #max_pool = tf.reduce_max(net, [1, 2])
        #mean_pool = tf.reduce_mean(net, [1, 2])
        #tf.concat(max_pool, mean_pool)
        shape = net.shape
        logger.info(f'pool shape: {shape}')
        flat = tf.reshape(net, [-1, shape[1] * shape[2] * shape[3]])
        net = tf.layers.dense(inputs=flat, units=1024, activation=tf.nn.relu)
        net = tf.layers.dropout(inputs=net, rate=0.6, training=mode == tf.estimator.ModeKeys.TRAIN)
        logits = tf.layers.dense(inputs=net, units=output_shape)
        return logits

    def estimator_spec(self, logits, labels, mode):
        predictions = {
            'logits': logits,
            'classes': tf.argmax(input=logits, axis=1),
            'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
        }

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        logger.info(f'logits shape: {logits.shape}')
        logger.info(f'labels shape: {labels.shape}')
        loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=labels, logits=logits)
        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
            train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
            export_outputs = {'x': tf.estimator.export.ClassificationOutput(scores=logits)}
            export_outputs = {'class': tf.estimator.export.ClassificationOutput(classes=tf.as_string(predictions['classes']))}
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op, export_outputs=export_outputs)

        if mode == tf.estimator.ModeKeys.EVAL:
            eval_metric_ops = self.metric_ops(labels, logits)
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

    def metric_ops(self, labels, logits):
        labels = tf.cast(labels, tf.int64)
        logits = tf.nn.sigmoid(logits)
        metric_ops = {}
        for k in range(1, 5):
            precision = tf.metrics.precision_at_k(labels, logits, k=k)
            recall = tf.metrics.recall_at_k(labels, logits, k=k)
            metric_ops[f'precision@k_{k}'] = precision
            metric_ops[f'recall@k_{k}'] = recall
            #metric_ops[f'f1_score@k_{k}'] = tf.div(tf.multiply(precision, recall), tf.add(precision,recall))
        thresholds = [x / 10.0 for x in range(1, 10)]
        precisions, prec_ops = tf.metrics.precision_at_thresholds(labels, logits, thresholds=thresholds)
        recalls, rec_ops = tf.metrics.recall_at_thresholds(labels, logits, thresholds=thresholds)
        for i, thresh in enumerate(thresholds):
            metric_ops[f'precision@thresh_{thresh}'] = (precisions[i], prec_ops[i])
            metric_ops[f'recall@thresh_{thresh}'] = (recalls[i], rec_ops[i])
            #metric_ops[f'f1_score@thresh_{thresh}'] = (precisions[i] * recalls[i]) / (precision[i] + recall[i])
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

    @functools.lru_cache()
    def estimator(self):
        model = tf.estimator.Estimator(model_fn=self.model_fn(), model_dir=f'/tmp/{self.name}')
        return model

    def input_fn(self, dataset, preprocessors=[], batch_size=5, n_epoch=1, buffer_size=10000):

        def f():
            ds = dataset
            for preprocessor in preprocessors:
                ds = preprocessor.apply(ds)
            ds = ds.shuffle(buffer_size=buffer_size)
            ds = ds.batch(batch_size)
            #ds = ds.repeat(n_epoch)
            iterator = ds.make_one_shot_iterator().get_next()
            return iterator

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
        # estimator.train(input_fn=self.input_fn(training, self.preprocessor), steps=50)
        # if validation:
        #     estimator.evaluate(input_fn=self.input_fn(validation, self.preprocessor), steps=50)
        train_spec = tf.estimator.TrainSpec(input_fn=self.input_fn(training, [processing.Preprocessor(self.preprocessor, flatten=True)]))
        eval_spec = tf.estimator.EvalSpec(input_fn=self.input_fn(validation, [processing.Preprocessor(self.preprocessor, flatten=True)]))
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    def predict(self, elements):
        tf.Graph().as_default()
        ds = tf.data.Dataset.from_tensor_slices(elements)
        logger.info(ds)
        estimator = self.estimator()
        preprocessors = [
            processing.PythonPreprocessor(self.prepreprocessor, output_dtypes=[tf.float32, tf.int32], output_shapes=([-1, 128],[-1])),
            processing.Preprocessor(self.preprocessor, flatten=True)
        ]
        return estimator.predict(self.input_fn(ds, preprocessors))

    # def save(self, base_dir):
    #     feature_spec = {'x': tf.FixedLenFeature([10, 128], dtype=tf.float32)}
    #     serving_input_receiver_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
    #     # def serving_input_receiver_fn():
    #     #     """An input receiver that expects a serialized tf.Example."""
    #     #     serialized_tf_example = tf.placeholder(dtype=tf.string,
    #     #                                            shape=[1],
    #     #                                            name='input_example_tensor')
    #     #     receiver_tensors = {'examples': serialized_tf_example}
    #     #     features = tf.parse_example(serialized_tf_example, feature_spec)
    #     #     return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)
    #     # serving_input_fn = tf.contrib.learn.utils.build_default_serving_input_fn(feature_spec)
    #     # receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(feature_spec)
    #     self.estimator().export_savedmodel(
    #         base_dir,
    #         serving_input_receiver_fn,
    #         assets_extra=None,
    #         as_text=False,
    #         checkpoint_path=None
    #     )

    @property
    def prepreprocessor(self):
        def f(x, y=None):
            x = feature.mel_spec(x, n_fft=self.n_fft, sr=self.sr)
            if y:
                return x, y
            return x
        return f

    @property
    def preprocessor(self):
        def f(x, y=None):
            if y is not None:
                y = tf.one_hot(y, len(self.metadata['label_dict']), dtype=tf.uint8)
                y = tf.reduce_sum(y, 0)
                X, Y = feature.split_spec_with_labels(
                    x,
                    y,
                    win_size=self.win_size,
                    hop_size=self.hop_size)
                return X, Y
            else:
                X = feature.split_spec_tf(
                    x,
                    win_size=self.win_size,
                    hop_size=self.hop_size)
                return X
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

        return util.wrap_tf_py_func(f, Tout=[self.metadata['data_struct'][0]['dtype'], self.metadata['data_struct'][1]['dtype']])
