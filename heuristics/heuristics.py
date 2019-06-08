import random as r

class Heuristic:
    def selected_pairs(self, pairs, point_ctn):
        pass

class RandomHeuristic(Heuristic):
    def selected_pairs(self, pairs, point_ctn):
        r.shuffle(pairs)
        return pairs