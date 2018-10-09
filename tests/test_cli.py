"""
test_cli
----------------------------------
Tests for `cli` module.
"""
import pytest
from click.testing import CliRunner

from tensionflow import cli


@pytest.fixture()
def runner():
    return CliRunner()


@pytest.fixture
def mocked_ConvPoolModel(mocker):
    mocked = mocker.patch('tensionflow.models.convpoolmodel.ConvPoolModel', autospec=True)
    mocked.train.return_value = 'trained!'
    return mocked


@pytest.fixture
def mocked_FmaDataset(mocker):
    mocked = mocker.patch('tensionflow.datasets.fma.FmaDataset', autospec=True)
    return mocked


def test_predict(mocker, runner, mocked_ConvPoolModel):
    result = runner.invoke(cli.cli, ['-v predict --model tmp'])
    print(result)


def test_cli(mocker, runner, mocked_ConvPoolModel, mocked_FmaDataset):
    result = runner.invoke(cli.train, ['--model ConvPoolModel --dataset FmaDataset'])
    print(result)
