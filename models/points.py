import numpy as np
from json import JSONEncoder

class Point:
    def __init__(self, x, y, index=0):
        self.x = x
        self.y = y
        self._index = index

    def dist(self, p):
        out = np.square(self.x - p.x) + np.square(self.y - p.y)
        return out

    def __lt__(self, other):
        return self._index < other._index

    def __eq__(self, other):
        return self._index == other._index

    def __repr__(self):
        return f"[X: {self.x}, Y: {self.y}]"


class KeyPoint(Point):
    def __init__(self, x, y, features, index=0):
        super().__init__(x, y)
        self._index = index
        self._features = features if isinstance(features, np.ndarray) else np.array(features)
        self._size = self._features.shape[0]

    def size(self):
        return self._size

    def feature_dist(self, keypoint):
        return np.linalg.norm(self._features - keypoint._features)

    def toJSON(self):
        point_dict = super().toJSON()
        point_dict['features'] = self._features
        return point_dict

    def __repr__(self):
        return f"[{self._index}] X: {self.x}, Y: {self.y}, Features: {self._features}"

    def __iter__(self):
        yield from {
            'x': self.x,
            'y': self.y,
            'features': self._features
        }.items()
