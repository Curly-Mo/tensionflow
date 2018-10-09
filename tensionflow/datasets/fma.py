import ast
import logging
import os

import pandas as pd

from tensionflow import datasets

logger = logging.getLogger(__name__)

DATA_DIR = '~/datasets/fma'


class FmaDataset(datasets.Dataset):
    def __init__(self, *args, size='small', data_dir=DATA_DIR, **kwargs):
        self.data_dir = data_dir
        self.size = size
        super().__init__(*args, **kwargs)

    def load_features_and_labels(self, split):
        X, Y, _ = self.get_dataset(split=split, size=self.size)
        return X, Y

    def load_fma_file(self, filepath):
        logger.info('Loading track metadata from %s', filepath)
        filename = os.path.basename(filepath)

        if 'features' in filename:
            return pd.read_csv(filepath, index_col=0, header=[0, 1, 2])

        if 'echonest' in filename:
            return pd.read_csv(filepath, index_col=0, header=[0, 1, 2])

        if 'genres' in filename:
            return pd.read_csv(filepath, index_col=0)

        if 'tracks' in filename:
            tracks = pd.read_csv(filepath, index_col=0, header=[0, 1])

            COLUMNS = [
                ('track', 'tags'),
                ('album', 'tags'),
                ('artist', 'tags'),
                ('track', 'genres'),
                ('track', 'genres_all'),
            ]
            for column in COLUMNS:
                tracks[column] = tracks[column].map(ast.literal_eval)

            COLUMNS = [
                ('track', 'date_created'),
                ('track', 'date_recorded'),
                ('album', 'date_created'),
                ('album', 'date_released'),
                ('artist', 'date_created'),
                ('artist', 'active_year_begin'),
                ('artist', 'active_year_end'),
            ]
            for column in COLUMNS:
                tracks[column] = pd.to_datetime(tracks[column])

            SUBSETS = ('small', 'medium', 'large')
            tracks['set', 'subset'] = tracks['set', 'subset'].astype(
                pd.api.types.CategoricalDtype(categories=SUBSETS, ordered=True)
            )

            COLUMNS = [('track', 'license'), ('artist', 'bio'), ('album', 'type'), ('album', 'information')]
            for column in COLUMNS:
                tracks[column] = tracks[column].astype('category')

            return tracks
        return None

    def path_from_id(self, track_id):
        prefix = os.path.join(self.data_dir, f'fma_{self.size}')
        prefix = os.path.expanduser(prefix)
        track_id = str(track_id).zfill(6)
        return os.path.join(prefix, track_id[0:3], track_id + '.mp3')

    def get_dataset(self, split='training', size='small'):
        logger.info('Loading dataset split=%s, size=%s', split, size)
        filepath = os.path.join(self.data_dir, 'fma_metadata/tracks.csv')
        tracks = self.load_fma_file(filepath)

        subset = tracks['set', 'subset'] == size
        if split == 'all':
            split = ('training', 'test', 'validation')
        else:
            split = (split,)
        split = tracks['set', 'split'].isin(split)
        # train = tracks['set', 'split'] == 'training'
        # val = tracks['set', 'split'] == 'validation'
        # test = tracks['set', 'split'] == 'test'
        genres_filepath = os.path.join(self.data_dir, 'fma_metadata/genres.csv')
        genres = self.load_fma_file(genres_filepath)
        labels = {row[0]: {'index': i, 'name': row[3], 'id': row[0]} for i, row in enumerate(genres.itertuples())}

        Y = tracks.loc[subset & split, ('track', 'genres_all')].values
        y_labels = []
        for y in Y:
            y_labels.append([labels[id]['name'] for id in y])
        X_ids = tracks.loc[subset & split].index.values
        X_files = [self.path_from_id(x) for x in X_ids]
        return X_files, y_labels, labels

    def get_dataset_single_genre(self, split='training'):
        filepath = os.path.join(self.data_dir, 'fma_metadata/tracks.csv')
        tracks = self.load_fma_file(filepath)

        small = tracks['set', 'subset'] <= 'small'
        split = tracks['set', 'split'] == split
        # train = tracks['set', 'split'] == 'training'
        # val = tracks['set', 'split'] == 'validation'
        # test = tracks['set', 'split'] == 'test'
        genres_filepath = os.path.join(self.data_dir, 'fma_metadata/genres.csv')
        genres = self.load_fma_file(genres_filepath)
        labels = {row[3]: {'index': i, 'name': row[3], 'id': row[0]} for i, row in enumerate(genres.itertuples())}

        Y = tracks.loc[small & split, ('track', 'genre_top')].values
        Y = [(v,) for v in Y]
        X_ids = tracks.loc[small & split].index.values
        X_files = [self.path_from_id(x) for x in X_ids]
        return X_files, Y, labels
