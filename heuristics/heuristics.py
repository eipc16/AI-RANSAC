import random as r


class Heuristic:
    def selected_pairs(self, pairs, points_ctn):
        pass


class RandomHeuristic(Heuristic):
    def selected_pairs(self, pairs, points_ctn):
        r.shuffle(pairs)
        return pairs[:points_ctn]
