import numpy as np
from json import JSONEncoder

class Point:
    def __init__(self, x, y):
        self._x = x;
        self._y = y;

    def dist(self, p):
        out = np.square(self._x - p._x) + np.square(self._y - p._y)
        return out

    def __repr__(self):
        return f"[X: {self._x}, Y: {self._y}]"

class KeyPoint(Point):
    def __init__(self, x, y, features, index = 0):
        super().__init__(x, y)
        self._index = index
        self._features = features if isinstance(features, np.ndarray) else np.array(features)
        self._size = self._features.shape[0]

    def size(self):
        return self._size

    def feature_dist(self, keypoint):
        feature_pairs = zip(self._features, keypoint._features)
        return np.sqrt(sum(np.square(x - y) for x, y in feature_pairs))

    def __repr__(self):
        return f"[{self._index}] X: {self._x}, Y: {self._y}, Features: {self._features}"