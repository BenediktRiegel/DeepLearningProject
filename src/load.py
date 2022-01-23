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

    def get_four_class_vector(self, valence, activation):
        if valence==0 and activation==0:
            return [1, 0, 0, 0]
        elif valence==1 and activation==0:
            return [0, 1, 0, 0]
        elif valence==0 and activation==1:
            return [0, 0, 1, 0]
        else:
            return [0, 0, 0, 1]

    def pad_sample(self, sample, maxdim):
        for i in range(maxdim - len(sample)):
            sample.append(np.zeros(26))
        sample = np.array(sample)
        sample = sample.reshape((sample.shape[0], sample.shape[1], 1))
        return sample

    def load_padded_X_y(self, path, four_class):
        j = self.load_json(path)
        X = []
        y = []
        maxdim = 1707
        for entry in j.values():
            X.append(self.pad_sample(entry['features'], maxdim))
            if four_class:
                y.append(self.get_four_class_vector(entry['valence'], entry['activation']))
            else:
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

    def load_ragged_X_y(self, path, four_class):
        j = self.load_json(path)
        X = []
        y = []
        maxdim = 1707
        for entry in j.values():
            X.append([[el/100.0 for el in lis] for lis in entry['features']])
            if four_class:
                y.append(self.get_four_class_vector(entry['valence'], entry['activation']))
            else:
                y.append([entry['valence'], entry['activation']])
        return tf.ragged.constant(X), tf.constant(y)

    def load_ragged_X(self, path):
        j = self.load_json(path)
        X = []
        maxdim = 1707
        for entry in j.values():
            X.append(entry['features'])
        return tf.ragged.constant(X)

    def load_training_data(self, mode='padded', four_class=False):
        if mode == 'padded':
            return self.load_padded_X_y(self.path / 'train.json', four_class)
        elif mode == 'ragged':
            return self.load_ragged_X_y(self.path / 'train.json', four_class)

    def load_dev_data(self, mode='padded'):
        if mode == 'padded':
            return self.load_padded_X(self.path / 'dev.json')
        elif mode == 'ragged':
            return self.load_ragged_X(self.path / 'dev.json')

    def load_short_training_data(self, four_class, mode='padded'):
        if mode == 'padded':
            return self.load_padded_X_y(self.path / 'short_train.json', four_class)
        elif mode == 'ragged':
            return self.load_ragged_X_y(self.path / 'short_train.json', four_class)

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
