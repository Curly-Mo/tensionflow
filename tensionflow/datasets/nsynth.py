import os
import json

METADATA_FILE = 'examples.json'
AUDIO_DIR = 'audio'
TRAIN_DIR = 'nsynth-train'
TEST_DIR = 'nsynth-test'
VALID_DIR = 'nsynth-valid'


class NSynth(object):
    def __init__(self,
                 data_dir=os.path.expanduser('~/datasets/nsynth'),
                 split='train'):
        if split == 'train':
            self.data_dir = os.path.join(data_dir, TRAIN_DIR)
        if split == 'test':
            self.data_dir = os.path.join(data_dir, TEST_DIR)
        if split == 'valid':
            self.data_dir = os.path.join(data_dir, VALID_DIR)
        self.audio_dir = os.path.join(self.data_dir, AUDIO_DIR)
        self.metadata_file = os.path.join(self.data_dir, METADATA_FILE)
        with open(self.metadata_file) as metadata_file:
            self.metadata = json.load(metadata_file)
        self.X, self.Y = zip(*[(os.path.join(self.audio_dir, f'{key}.wav'),
                                meta) for key, meta in self.metadata.items()])
