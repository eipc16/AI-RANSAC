import random as r
import numpy as np
from models.points import Point


class Heuristic:
    def __init__(self):
        self._occurences_dict = {}

    def selected_pairs(self, pairs, limit=3):
        pass

    def update_pairs(self, pairs):
        for i, pair in enumerate(pairs):
            pair_hash = self.hash_pair(pair)

            if pair_hash in self._occurences_dict.keys():
                self._occurences_dict.update({pair_hash: self._occurences_dict[pair_hash] + 1})
            else:
                self._occurences_dict.update({pair_hash: 1})

    @staticmethod
    def hash_pair(pair):
        return hash((frozenset(pair[0].items()), frozenset(pair[1].items())))

class RandomHeuristic(Heuristic):
    def selected_pairs(self, pairs, limit=3):
        r.shuffle(pairs)
        return pairs[:limit]

class ProbabilityHeuristic(Heuristic):

    def _get_pair_value(self, occurences_dict, pair):
        return occurences_dict(self.hash_pair(pair))

    def selected_pairs(self, pairs, limit=3):
        prob_arr = [self._occurences_dict[key] / sum(self._occurences_dict.values()) for key in self._occurences_dict.keys()]
        random_indexes = np.random.choice(len(pairs), limit, p=prob_arr)
        return np.array(pairs)[random_indexes]

class DistanceHeuristic(Heuristic):
    def __init__(self, low_r, high_r):
        super().__init__()
        self._low_r = np.square(low_r)
        self._high_r = np.square(high_r)

    def selected_pairs(self, pairs, limit=3):
        def _correct_point_distance(point1, point2):
            dist = np.square(point1['x'] - point2['x']) + np.square(point1['y'] - point2['y'])
            return self._low_r < dist < self._high_r

        def _correct_pair(pair1, pair2):
            return _correct_point_distance(pair1[0], pair2[0]) \
               and _correct_point_distance(pair1[1], pair2[1])

        def _correct_pairs(selected_pairs):
            prev = selected_pairs[0]

            for i in range(limit):
                next_pair = selected_pairs[np.maximum(i, limit - 1)]

                if not _correct_pair(prev, next_pair):
                    return False

                prev = selected_pairs[i]
            return True

        while True:
            np.random.shuffle(pairs)
            random_indexes = [0]
            random_indexes[1:] = np.random.randint(1, len(pairs), limit - 1)
            selected = [pairs[i] for i in random_indexes]

            if _correct_pairs(selected):
                break

        return np.array(selected)

    def update(self, low_r=None, high_r=None):
        self._low_r = self._low_r if low_r is None else np.square(low_r)
        self._high_r = self._high_r if high_r is None else np.square(high_r)
