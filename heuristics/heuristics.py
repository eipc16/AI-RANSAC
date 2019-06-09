import random as r


class Heuristic:
    def selected_pairs(self, pairs, limit=3):
        pass


class RandomHeuristic(Heuristic):
    def selected_pairs(self, pairs, limit=3):
        r.shuffle(pairs)
        return pairs[:limit]
