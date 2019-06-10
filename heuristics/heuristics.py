import random as r
import numpy as np
from multiprocessing import Pool
from functools import partial

class Heuristic:
    def __init__(self):
        self._occurences_dict = {}

    def selected_pairs(self, pairs, limit=3):
        pass

    def update_pairs(self, pairs, value=1, increments=None):
        def _get_new_value(pos, val, inc):
            return self._occurences_dict[pos] + inc if inc is not None else val

        for pair in pairs:
            pair_hash = self.hash_pair(pair)
            self._occurences_dict[pair_hash] = _get_new_value(pair_hash, value, increments)

    @staticmethod
    def hash_pair(pair):
        return hash((frozenset(pair[0].items()), frozenset(pair[1].items())))

    def _get_pair_value(self, pair):
        return self._occurences_dict[self.hash_pair(pair)]

class RandomHeuristic(Heuristic):
    def selected_pairs(self, pairs, limit=3):
        return np.random.choice(pairs, limit)


class ProbabilityHeuristic(Heuristic):
    def selected_pairs(self, pairs, limit=3):
        prob_arr = np.array([self._occurences_dict[key] / sum(self._occurences_dict.values()) for key in self._occurences_dict.keys()])
        random_indexes = np.random.choice(pairs.shape[0], limit, p=prob_arr)
        return pairs[random_indexes]


class ReductionHeuristic(Heuristic):
    def selected_pairs(self, pairs, limit=3):
        selected = []
        random_indexes = np.random.choice(len(pairs), limit)

        for i in random_indexes:
            value = self._get_pair_value(pairs[i])

            if value > 0:
                selected.append(i)

            if len(selected) >= limit:
                break

        while len(selected) < limit:
            random = np.random.randint(0, len(pairs))
            if random not in selected:
                selected.append(random)

        return pairs[selected]


class NeighbourHeuristic(Heuristic):
    def __init__(self):
        super().__init__()
        self._pool = Pool()

    @staticmethod
    def _calc_distance(point1, point2):
        return np.square(point1['x'] - point2['x']) + np.square(point1['y'] - point2['y'])

    def calc_neighbours(self, point, points, r):
        distances = np.sum(np.linalg.norm(points - point))
        limit = np.square(r)
        return np.count_nonzero(distances < limit)

    def selected_pairs(self, pairs, limit=3):
        np.random.shuffle(pairs)
        coords = np.array([[[p[0]['x'], p[0]['y']], [p[1]['x'], p[1]['y']]] for p in pairs])

        selected = []
        for i, pair in enumerate(coords):
            left = self.calc_neighbours(pair[0], coords[:, 0], 5)
            right = self.calc_neighbours(pair[1], coords[:, 1], 5)

            if left > 5 and right > 5:
                selected.append(i)

            if len(selected) >= limit:
                break

        while len(selected) < limit:
            random = np.random.randint(0, len(pairs))
            if random not in selected:
                selected.append(random)

        return pairs[selected]


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
