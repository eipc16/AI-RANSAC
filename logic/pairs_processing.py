import operator as op
import functools as f
import numpy as np

from models.points import Point
from utils.time_utils import get_execution_time


class PairProcessor:
    @get_execution_time
    def consistent_pairs(self, pairs, neighbours_limit, threshold):
        def _neighbours_count(pair):
            neighbours_of_first = \
                self._neighbours(pair[0],
                                 list(filter(f.partial(op.ne, pair), pairs)), neighbours_limit)
            neighbours_of_second = \
                self._neighbours(pair[1],
                                 list(map(lambda x: x[1], filter(f.partial(op.ne, pair), pairs))), neighbours_limit)

            return len(list(filter(lambda x: x[1] in neighbours_of_second, neighbours_of_first)))

        def _in_threshold(x):
            return float(_neighbours_count(x) / neighbours_limit >= threshold)

        filtered_points = np.array(list(filter(lambda p : _in_threshold(p), pairs)))
        return np.array([(Point(k['_index'], k['x'], k['y']), Point(k2['_index'], k2['x'], k2['y'])) for k, k2 in filtered_points])

    @staticmethod
    def _neighbours(point, points, neighbours_limit):
        if len(points) < 0:
            return np.array([])

        sec_img_points = np.array(points)
        if isinstance(points[0], list):
            sec_img_points = np.array([t[0] for t in points])

        point_coords = np.array([[point['x'], point['y']]])
        points_coords = np.array([[p['x'], p['y']] for p in sec_img_points])
        dist = np.sum((points_coords - point_coords) ** 2, axis=1)
        indices = np.argsort(dist)[:neighbours_limit]
        return np.array(points)[indices]
