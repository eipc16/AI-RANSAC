import numpy as np
from functools import partial
from multiprocessing import Pool
from utils.time_utils import get_execution_time


class Ransac:
    def start(self, pairs, max_error, iterations, transformation, p=0.0, w=0.0):
        model = self._get_best_model(pairs, max_error, iterations, transformation, p, w)

        return list(filter(lambda p: self._model_error(p, model) < max_error, pairs))

    @get_execution_time
    def _get_best_model(self, pairs, max_error, iterations, transformation, p, w):
        best_model, best_score = None, 0
        pool = Pool()

        estimated_iterations = \
            np.log(1 - p) \
            / (np.log(1 - np.power(w, transformation.get_points_cnt())))

        iterations = int(np.minimum(iterations, estimated_iterations) \
            if estimated_iterations > 0 else iterations)

        print(iterations)

        for i in range(iterations):
            model, score = None, 0
            while model is None:
                model = transformation.get_model(pairs)

            copier = partial(self._model_error, model=model)
            errors = np.array(pool.map(copier, pairs))
            score = np.sum(errors < max_error)

            if score > best_score:
                print(f'Found new best [Score: {score}] [Prev: {best_score}] [Iteration: {i}]')
                best_score = score
                best_model = model

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

