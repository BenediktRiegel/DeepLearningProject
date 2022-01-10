import json
from pathlib import Path
from abc import abstractmethod
import tensorflow as tf
import numpy as np


class Loader:
    def __init__(self, path: str):
        self.path = Path(path)

    def load_json(self, path):
        return json.load(path.open('r'))

    @abstractmethod
    def load_training_data(self):
        pass

    @abstractmethod
    def load_dev_data(self):
        pass


class serLoader(Loader):

    def __init__(self):
        super().__init__('../data/SER')

    def pad_sample(self, sample, maxdim):
        for i in range(maxdim - len(sample)):
            sample.append(np.zeros(26))
        sample = np.array(sample)
        sample = sample.reshape((sample.shape[0], sample.shape[1], 1))
        return sample

    def load_padded_X_y(self, path):
        j = self.load_json(path)
        X = []
        y = []
        maxdim = 1707
        for entry in j.values():
            X.append(self.pad_sample(entry['features'], maxdim))
            y.append([entry['valence'], entry['activation']])
        X = np.array(X)
        y = np.array(y)
        return X, y

    def load_padded_X(self, path):
        j = self.load_json(path)
        X = []
        maxdim = 1707
        for entry in j.values():
            X.append(self.pad_sample(entry['features'], maxdim))
        return np.array(X)

    def load_ragged_X_y(self, path):
        j = self.load_json(path)
        X = []
        y = []
        maxdim = 1707
        for entry in j.values():
            X.append(entry['features'])
            y.append([entry['valence'], entry['activation']])
        return tf.ragged.constant(X), tf.constant(y)

    def load_ragged_X(self, path):
        j = self.load_json(path)
        X = []
        maxdim = 1707
        for entry in j.values():
            X.append(entry['features'])
        return tf.ragged.constant(X)

    def load_training_data(self, mode='padded'):
        if mode == 'padded':
            return self.load_padded_X_y(self.path / 'train.json')
        elif mode == 'ragged':
            return self.load_ragged_X_y(self.path / 'train.json')

    def load_dev_data(self, mode='padded'):
        if mode == 'padded':
            return self.load_padded_X(self.path / 'dev.json')
        elif mode == 'ragged':
            return self.load_ragged_X(self.path / 'dev.json')

    def load_short_training_data(self, mode='padded'):
        if mode == 'padded':
            return self.load_padded_X_y(self.path / 'short_train.json')
        elif mode == 'ragged':
            return self.load_ragged_X_y(self.path / 'short_train.json')

    def create_short_version(self, num_entries):
        j = self.load_json(self.path / 'train.json')
        j_ = dict()
        keys = list(j.keys())
        for i in range(num_entries):
            j_[keys[i]] = j[keys[i]]
        path = self.path / 'short_train.json'
        with path.open('w') as f:
            json.dump(j_, f)
            f.close()
