"""Command line interface for tensionflow"""
import logging
import sys

import click

from tensionflow import models
from tensionflow import datasets
from tensionflow import util
from tensionflow.datasets import fma
from tensionflow.util import cli_util
from tensionflow.models.basemodel import BaseModel

logger = logging.getLogger(__name__)
# tf.logging._logger.propagate = False


ALL_MODELS = util.get_all_subclasses(models.Model)
ALL_DATASETS = util.get_all_subclasses(datasets.Dataset)


@click.group()
@click.option('-v', '--verbose', count=True)
def cli(verbose):
    initLogging(verbose)


@cli.command()
@click.option('-m', '--model', cls=cli_util.Mutex, mutex_with=['saved_model'], type=click.Choice(ALL_MODELS.keys()))
@click.option('-d', '--dataset', cls=cli_util.Mutex, mutex_with=['saved_dataset'], type=click.Choice(ALL_DATASETS.keys()))
@click.option('--saved_model', cls=cli_util.Mutex, mutex_with=['model'], type=click.Path(exists=True))
@click.option('--saved_dataset', cls=cli_util.Mutex, mutex_with=['dataset'], type=click.Path(exists=True))
def train(model, dataset, saved_model, saved_dataset):
    """Train a model with a given dataset"""
    if model:
        m = ALL_MODELS[model]()
    else:
        m = models.Model.load(saved_model)
    if dataset:
        ds = ALL_DATASETS[dataset]()
    else:
        ds = datasets.Dataset(filepath=saved_dataset)
    click.echo(f'Training {m.__class__.__name__} with {ds.__class__.__name__}')
    try:
        m.train(ds)
    except:
        click.echo('Abort! Save the model now')
        print(sys.exc_info()[0])
        model.save()


@cli.command()
@click.argument('model', type=click.Path(exists=True))
@click.argument('audiofile', nargs=-1, type=click.Path(exists=True))
def predict(model, audiofile):
    """Train a model with a given dataset"""
    m = models.Model.load(model)
    audiofiles = list(audiofile)
    click.echo(f'Predicting {audiofiles} using trained {m.__class__.__name__}, {model}')
    predictions = m.predict(audiofiles)
    for prediction in predictions:
        click.echo(prediction)


def initLogging(verbosity):
    """Setup logging with a given verbosity level"""
    logging.basicConfig()
    if verbosity == 0:
        logging.root.setLevel(logging.WARN)
    if verbosity == 1:
        logging.root.setLevel(logging.INFO)
    if verbosity > 0:
        logging.root.setLevel(logging.DEBUG)
