import operator as op
import functools as f

class PairProcessor:
    def consistent_pairs(self, pairs, neighbours_limit, threshold):
        def _neighbours_count(pair):
            neighboursOfFirst = self._neighbours(pair[0], list(filter(f.partial(op.ne, pair), pairs)), neighbours_limit)
            neighboursOfSecond = self._neighbours(pair[1], list(map(lambda x: x[1], filter(f.partial(op.ne, pair), pairs))), neighbours_limit)
            return len(list(filter(lambda x: x[1] in neighboursOfSecond, neighboursOfFirst)))

        def _in_threshold(x):
            return float(_neighbours_count(x)) / neighbours_limit >= threshold

        return list(filter(lambda x: _in_threshold(x), pairs))


    def _neighbours(self, point, points, neighbours_limit):
        def _unpack(p):
            return p[0] if isinstance(p, tuple) else p

        distances = list(map(lambda p: (p, point.dist(_unpack(p))), points))
        distances = sorted(distances, key=lambda x: x[1])
        return list(map(op.itemgetter(0), distances[:neighbours_limit]))


from models.points import Point

x = [(Point(1, 122), Point(2, 400)), (Point(4000, 1), Point(1, 5)), (Point(4, 3), Point(2, 4)), (Point(4, 1), Point(2, 3)), (Point(2, 0), Point(1, 666))]

p_x = Point(4, 3)
p_y = Point(2, 4)

processor = PairProcessor()

pairs = processor.consistent_pairs(x, 4, 0.0)
print(pairs)

pairs = processor.consistent_pairs(x, 4, 0.1)
print(pairs)

pairs = processor.consistent_pairs(x, 4, 2.0)
print(pairs)