import numpy as np
from functools import partial
from multiprocessing import Pool

class Ransac:
    def start(self, pairs, max_error, iterations, transformation):
        model = self._get_best_model(pairs, max_error, iterations, transformation)

        return list(filter(lambda p: self._model_error(p, model) < max_error, pairs))

    def _get_best_model(self, pairs, max_error, iterations, transformation):
        best_model, best_score = None, 0
        p = Pool()

        for i in range(iterations):
            print(f'Ransac iteration {i + 1}')
            model, score = None, 0
            while model is None:
                model = transformation.get_model(pairs)
            
            copier = partial(self._model_error, model=model)
            errors = np.array(p.map(copier, pairs))
            score = np.where(errors < max_error)[0].shape[0]
            
            if score > best_score:
                best_score = score
                best_model = model

        return best_model

    @staticmethod
    def _model_error(data, model):
        x, y = data[0]['x'], data[1]['y']

        sec_matrix = np.array([x * 1.0, y * 1.0, 1.0], dtype=float)
        result = model @ sec_matrix.T

        t = result[2]
        u, v = result[0] / t, result[1] / t

        real_u, real_v = data[1]['x'], data[1]['y']

        return np.sqrt(np.square(u - real_u) + np.square(v - real_v))