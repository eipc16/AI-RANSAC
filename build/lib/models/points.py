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
    def __init__(self, x, y, features):
        super().__init__(x, y)
        self._features = features if isinstance(features, np.ndarray) else np.array(features)
        self._size = self._features.shape[0]

    def size(self):
        return self._size

    def __repr__(self):
        return f"X: {self._x}, Y: {self._y}, Features: {self._features}"