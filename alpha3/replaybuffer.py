import numpy as np

class ReplayBuffer:
    def __init__(self, max_size, features_shape, label_shape):
        self._features = np.zeros((max_size, *features_shape), dtype='float32')
        self._labels = np.zeros((max_size, *label_shape), dtype='float32')
        self._max_size = max_size
        self._size = 0
        self._oldest_index = 0

    def insert(self, features, label):
        self._size = min(self._size + 1, self._max_size)

        self._features[self._oldest_index, :] = features
        self._labels[self._oldest_index, :] = label

        self._oldest_index += 1
        self._oldest_index %= self._max_size

    def sample(self, size):
        size = min(size, self._size)
        indices = np.random.choice(self._size, size, replace=False)
        return (self._features[indices, :], self._labels[indices, :])

    def __len__(self):
        return self._size
