import logging
import functools
import os
import shutil
import pickle  # nosec
import tempfile
import abc
import time

import tensorflow as tf

# from tensorflow.contrib import predictor

from tensionflow import datasets
from tensionflow import processing

logger = logging.getLogger(__name__)

# tf.logging._logger.propagate = False


class Model(abc.ABC):
    def __init__(
        self, name='AbstractModel', save_dir=None, batch_size=32, eval_examples=3000, eval_every_n_examples=100_000
    ):
        self.name = name
        self.save_dir = save_dir
        self.batch_size = batch_size
        # Measture steps on the example level, not the batch level
        self.eval_steps = None if eval_examples is None else eval_examples // batch_size
        self.eval_every_n_iter = None if eval_every_n_examples is None else eval_every_n_examples // batch_size
        if self.save_dir is None:
            self.save_dir = f'{name}.{int(time.time())}'
        self.metadata = {'class': self.__class__}
        self.sess = tf.Session()
        # self.load()
        if not hasattr(self, 'estimator'):
            model_dir = tempfile.mkdtemp(prefix='tensionflow.')
            self.estimator = tf.estimator.Estimator(model_fn=self.model_fn(), model_dir=model_dir)

    @staticmethod
    def load(saved_path=None):
        model_dir = tempfile.mkdtemp(prefix='tensionflow.')
        if saved_path:
            logger.info('Loading metadata from %s', Model.metafile(saved_path))
            with open(Model.metafile(saved_path), 'rb') as handle:
                metadata = pickle.load(handle)  # nosec
                logger.debug('Loaded metadata: %s', metadata)
                model = metadata['class']()
                model.metadata = metadata
            os.rmdir(model_dir)
            logger.info('Copying %s to %s', saved_path, model_dir)
            shutil.copytree(saved_path, model_dir)
        model.estimator = tf.estimator.Estimator(model_fn=model.model_fn(), model_dir=model_dir)
        return model

    @abc.abstractmethod
    def network(self, features, output_shape, mode):
        pass

    @abc.abstractmethod
    def estimator_spec(self, logits, labels, mode):
        pass

    @abc.abstractmethod
    def metric_ops(self, labels, logits):
        pass

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

    def input_fn(self, dataset, preprocessors=(), n_epoch=None, buffer_size=100_000):
        def f():
            ds = dataset
            for preprocessor in preprocessors:
                print(preprocessor.func)
                ds = preprocessor.apply(ds)
            ds = ds.shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True, seed=None)
            ds = ds.batch(self.batch_size)
            ds = ds.repeat(n_epoch)
            iterator = ds.make_one_shot_iterator().get_next()
            print(iterator)
            features, labels = iterator
            return features, labels
            # try:
            #     features, labels = iterator
            #     features = {'features': features}
            #     return features, labels
            # except:
            #     features = {'features': iterator}
            #     return features

        return f

    def train(self, dataset):
        logger.info('Training model in tmp location: %s', self.estimator.model_dir)
        self.metadata.update(dataset.meta)
        training = validation = None
        if isinstance(dataset, str):
            # If dataset is a path to a saved dataset, load it
            dataset = datasets.Dataset(dataset, ['training'], functools.partial(self.prepreprocessor))
        if isinstance(dataset, datasets.Dataset):
            training = dataset.splits['training']
            validation = dataset.splits['validation']
        estimator = self.estimator
        # estimator.train(input_fn=self.input_fn(training, self.preprocessor), steps=50)
        # if validation:
        #     estimator.evaluate(input_fn=self.input_fn(validation, self.preprocessor), steps=50)

        # train_spec = tf.estimator.TrainSpec(input_fn=self.input_fn(training, self.preprocessors))
        # eval_spec = tf.estimator.EvalSpec(
        #     input_fn=self.input_fn(validation, self.preprocessors), start_delay_secs=10, throttle_secs=150
        # )
        # tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

        train_input_fn = self.input_fn(training, self.preprocessors)
        eval_input_fn = self.input_fn(validation, self.preprocessors, n_epoch=1)
        evaluator = tf.contrib.estimator.InMemoryEvaluatorHook(
            estimator, eval_input_fn, steps=self.eval_steps, name=None, every_n_iter=self.eval_every_n_iter
        )
        estimator.train(train_input_fn, hooks=[evaluator])

    def predict(self, elements):
        tf.Graph().as_default()
        ds = tf.data.Dataset.from_tensor_slices(elements)
        logger.debug(ds)
        estimator = self.estimator
        preprocessors = self.prepreprocessors + self.preprocessors
        return estimator.predict(self.input_fn(ds, preprocessors, n_epoch=1))

    def save(self, output_dir='saved_models', force=False):
        dst = os.path.join(output_dir, self.save_dir)
        try:
            logger.info('Copying %s to %s', self.estimator.model_dir, dst)
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
            inputs = {'features': tf.placeholder(shape=[None, None, 128], dtype=tf.float32)}
            return tf.estimator.export.ServingInputReceiver(inputs, inputs)

        # def serving_input_receiver_fn():
        #     features = tf.placeholder(dtype=tf.float32,
        #                              shape=[None, None, 128],
        #                              name='input_example_tensor')
        #     feature_spec = {'features': features}
        #     return tf.estimator.export.ServingInputReceiver(feature_spec, feature_spec)
        # serving_input_receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(feature_spec)
        self.estimator.export_savedmodel(
            base_dir, serving_input_receiver_fn, assets_extra=None, as_text=False, checkpoint_path=None
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
        return [processing.Preprocessor(functools.partial(self.preprocessor), flatten=True)]

    @abc.abstractmethod
    def prepreprocessor(self, x, y=None):
        pass

    @abc.abstractmethod
    def preprocessor(self, x, y=None):
        pass

    @staticmethod
    def metafile(directory):
        return os.path.join(directory, 'metadata')
