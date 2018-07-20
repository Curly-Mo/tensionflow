import os
import json

from tensionflow import datasets

METADATA_FILE = 'examples.json'
AUDIO_DIR = 'audio'
TRAIN_DIR = 'nsynth-train'
TEST_DIR = 'nsynth-test'
VALID_DIR = 'nsynth-valid'


class NSynth(datasets.Dataset):
    def __init__(self, *args, data_dir=os.path.expanduser('~/datasets/nsynth'), **kwargs):
        self.data_dir = data_dir
        super().__init__(*args, **kwargs)

    def load_features_and_labels(self, split):
        data_dir = self.split_dirs[split]
        audio_dir = os.path.join(data_dir, AUDIO_DIR)
        metadata_file = os.path.join(data_dir, METADATA_FILE)
        with open(metadata_file) as metadata_file:
            self.metadata = json.load(metadata_file)
        X, Y = zip(*[(os.path.join(audio_dir, f'{key}.wav'), meta) for key, meta in self.metadata.items()])
        return X, Y

    @property
    def split_dirs(self):
        return {
            'train': os.path.join(self.data_dir, TRAIN_DIR),
            'test': os.path.join(self.data_dir, TEST_DIR),
            'validation': os.path.join(self.data_dir, VALID_DIR),
        }
