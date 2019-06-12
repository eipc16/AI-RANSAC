import numpy as np
from functools import partial
from multiprocessing import Pool
from utils.time_utils import get_execution_time
from heuristics.heuristics import RandomHeuristic, ProbabilityHeuristic, ReductionHeuristic
from transformations.transform import AffineTransformation
from time import time
from tqdm import tqdm

class Ransac:
    def start(self, pairs, max_error, iterations, new_pairs, transformation=AffineTransformation(), heuristic=RandomHeuristic(),
              p=None, w=0.0):
        model = self._get_best_model(np.array(pairs), max_error, iterations, transformation, heuristic, p, w)

        if model is None:
            return list()

        return list(filter(lambda p: self._model_error(p, model) < max_error, pairs))

    @get_execution_time
    def _get_best_model(self, pairs, max_error, iterations, transformation, heuristic, p, w):
        time_start = time()
        best_model, best_score, worst_score = None, 0, len(pairs) + 1
        pool = Pool()

        if p is not None:
            numerator = np.log(1 - p)
            denominator = np.log(1 - np.power(w, transformation.get_points_cnt()))
            estimated_iterations = numerator / (denominator if denominator != 0 else 0.0000001)

            iterations = int(round(np.minimum(iterations, estimated_iterations) if estimated_iterations > 0 else iterations))

        transformation.set_heuristic(heuristic)

        if isinstance(heuristic, ProbabilityHeuristic) \
                or isinstance(heuristic, ReductionHeuristic):
            transformation.update_occurences(pairs, new_value=1)

        prev_score = 0

        for i in tqdm(range(iterations)):
            model, score, selected_pairs = None, 0, len(pairs)
            while model is None:
                model, selected_pairs = transformation.get_model(pairs)

            copier = partial(self._model_error, model=model)
            errors = np.array(pool.map(copier, pairs))
            score = np.sum(errors < max_error)

            if score > best_score:
                tqdm.write(f'Found new best [Score: {score}] [Prev: {best_score}] '
                           f'[Iteration: {i}] [Time: {(time() - time_start):4f}s]')

                if isinstance(heuristic, ProbabilityHeuristic):
                    transformation.update_occurences(selected_pairs, new_increments=10)
                if isinstance(heuristic, ReductionHeuristic):
                    transformation.update_occurences(selected_pairs, new_value=1)

                best_score = score
                best_model = model

            if isinstance(heuristic, ReductionHeuristic) and i > 1:
                if score <= worst_score:
                    # tqdm.write(f'Found new worst [Score: {score}] '
                    #            f'[Prev: {worst_score}] [Best: {best_score}] [Iteration: {i}]')
                    transformation.update_occurences(selected_pairs, new_value=0)
                    worst_score = score
                elif score > prev_score:
                    transformation.update_occurences(selected_pairs, 1)

            prev_score = score

        return best_model

    @staticmethod
    def _model_error(data, model):
        x, y = data[0]['x'], data[0]['y']

        sec_matrix = np.array([x * 1.0, y * 1.0, 1.0], dtype=float)
        result = model @ sec_matrix.T

        t = result[2]
        u, v = result[0] / t, result[1] / t

        real_u, real_v = data[1]['x'], data[1]['y']

        return np.sqrt(np.square(u - real_u) + np.square(v - real_v))
