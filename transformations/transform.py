import numpy as np
from heuristics.heuristics import Heuristic, ProbabilityHeuristic

class Transformation:
    def __init__(self, points_cnt):
        self._points_cnt = points_cnt
        self._heuristic = None

    def get_model(self, pairs):
        pass

    def _result_vector(self, vector):
        pass

    def _model(self, matrix, vector):
        try:
            inv_matrix = np.linalg.inv(matrix)
            mult_res = inv_matrix @ vector
            return self._result_vector(mult_res.flatten())
        except np.linalg.LinAlgError:
            return None

    def get_points_cnt(self):
        return self._points_cnt

    def update_occurences(self, pairs, new_value=None, new_increments=None):
        self._heuristic.update_pairs(pairs, value=new_value, increments=new_increments)

    def set_heuristic(self, heuristic):
        if isinstance(heuristic, Heuristic):
            self._heuristic = heuristic
        else:
            raise Exception('Heuristic has to extend Heuristic class')


class AffineTransformation(Transformation):
    def __init__(self):
        super().__init__(3)
    
    def get_model(self, pairs):
        selected = self._heuristic.selected_pairs(pairs, limit=self.get_points_cnt())
        
        matrix = np.array([
            [selected[0][0]['x'], selected[0][0]['y'], 1.0, 0.0, 0.0, 0.0],
            [selected[1][0]['x'], selected[1][0]['y'], 1.0, 0.0, 0.0, 0.0],
            [selected[2][0]['x'], selected[2][0]['y'], 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, selected[0][0]['x'], selected[0][0]['y'], 1.0],
            [0.0, 0.0, 0.0, selected[1][0]['x'], selected[1][0]['y'], 1.0],
            [0.0, 0.0, 0.0, selected[2][0]['x'], selected[2][0]['y'], 1.0]
        ])

        vector = np.array([
            selected[0][1]['x'],
            selected[1][1]['x'],
            selected[2][1]['x'],
            selected[0][1]['y'],
            selected[1][1]['y'],
            selected[2][1]['y'],
        ])

        return self._model(matrix, vector.T), selected

    def _result_vector(self, vector):
        return np.array([
            [vector[0], vector[1], vector[2]],
            [vector[3], vector[4], vector[5]],
            [0.0, 0.0, 1.0]
        ])


class PerspectiveTransformation(Transformation):
    def __init__(self):
        super().__init__(4)

    def get_model(self, pairs):
        selected = self._heuristic.selected_pairs(pairs, limit=self.get_points_cnt())

        matrix = np.array([
            [selected[0][0]['x'], selected[0][0]['y'], 1.0, 0.0, 0.0, 0.0, -selected[0][1]['x'] * selected[0][0]['x'], -selected[0][1]['y'] * selected[0][0]['x']],
            [selected[1][0]['x'], selected[1][0]['y'], 1.0, 0.0, 0.0, 0.0, -selected[1][1]['x'] * selected[1][0]['x'], -selected[1][1]['y'] * selected[1][0]['x']],
            [selected[2][0]['x'], selected[2][0]['y'], 1.0, 0.0, 0.0, 0.0, -selected[2][1]['x'] * selected[2][0]['x'], -selected[2][1]['y'] * selected[2][0]['x']],
            [selected[3][0]['x'], selected[3][0]['y'], 1.0, 0.0, 0.0, 0.0, -selected[3][1]['x'] * selected[3][0]['x'], -selected[3][1]['y'] * selected[3][0]['x']],
            [0.0, 0.0, 0.0, selected[0][0]['x'], selected[0][0]['y'], 1.0, -selected[0][1]['y'] * selected[0][0]['x'], -selected[0][1]['y'] * selected[0][0]['y']],
            [0.0, 0.0, 0.0, selected[1][0]['x'], selected[1][0]['y'], 1.0, -selected[1][1]['y'] * selected[1][0]['x'], -selected[1][1]['y'] * selected[1][0]['y']],
            [0.0, 0.0, 0.0, selected[2][0]['x'], selected[2][0]['y'], 1.0, -selected[2][1]['y'] * selected[2][0]['x'], -selected[2][1]['y'] * selected[2][0]['y']],
            [0.0, 0.0, 0.0, selected[3][0]['x'], selected[3][0]['y'], 1.0, -selected[3][1]['y'] * selected[3][0]['x'], -selected[3][1]['y'] * selected[3][0]['y']]
        ])

        vector = np.array([
            selected[0][1]['x'],
            selected[1][1]['x'],
            selected[2][1]['x'],
            selected[3][1]['x'],
            selected[0][1]['y'],
            selected[1][1]['y'],
            selected[2][1]['y'],
            selected[3][1]['y'],
        ])

        return self._model(matrix, vector.T), selected

    def _result_vector(self, vector):
        return np.array([
            [vector[0], vector[1], vector[2]],
            [vector[3], vector[4], vector[5]],
            [vector[6], vector[7], 1.0]
        ])
