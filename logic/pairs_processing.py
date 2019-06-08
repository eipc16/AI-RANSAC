import operator as op
import functools as f
from multiprocessing import Pool

from utils.time_utils import get_execution_time

class PairProcessor:
    @get_execution_time
    def consistent_pairs(self, pairs, neighbours_limit, threshold):
        def _neighbours_count(pair):
            neighboursOfFirst = self._neighbours(pair[0], list(filter(f.partial(op.ne, pair), pairs)), neighbours_limit)
            neighboursOfSecond = self._neighbours(pair[1], list(map(lambda x: x[1], filter(f.partial(op.ne, pair), pairs))), neighbours_limit)
            return len(list(filter(lambda x: x[1] in neighboursOfSecond, neighboursOfFirst)))

        def _in_threshold(x):
            return float(_neighbours_count(x)) / neighbours_limit >= threshold

        return list(filter(lambda x: _in_threshold(x), pairs))


    def _neighbours(self, point, points, neighbours_limit):
        distances = list(map(lambda x: self.get_distance_tuple(x, point), points))
        distances = sorted(distances, key=lambda x: x[1])
        return list(map(op.itemgetter(0), distances[:neighbours_limit]))

    @staticmethod
    def get_distance_tuple(target, source):
        def _unpack(p):
            return p[0] if isinstance(p, tuple) else p

        return (target, source.dist(_unpack(target)))