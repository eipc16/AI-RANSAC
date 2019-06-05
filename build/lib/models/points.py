import numpy as np

class Point:
    def __init__(self, x, y):
        self._x = x;
        self._y = y;

    def dist(self, p):
        return np.square(self._x - p._x) + np.square(self._y - p._y)

class KeyPoint(Point):
    def __init__(self, x, y, features):
        super().__init__(x, y)
        self._features = features
        self._size = features.shape[0]

    def size(self):
        return self._size