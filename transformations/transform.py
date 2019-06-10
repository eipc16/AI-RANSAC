import numpy as np

class Transformation:
    def __init__(self, points_cnt):
        self._points_cnt = points_cnt

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


class AffineTransformation(Transformation):
    def __init__(self, heuristic):
        super().__init__(3)
        self._heuristic = heuristic
    
    def get_model(self, pairs):
        selectedPairs = self._heuristic.selected_pairs(pairs, limit=self.get_points_cnt())
        
        matrix = np.array([
            [selectedPairs[0][0]['x'], selectedPairs[0][0]['y'], 1.0, 0.0, 0.0, 0.0],
            [selectedPairs[1][0]['x'], selectedPairs[1][0]['y'], 1.0, 0.0, 0.0, 0.0],
            [selectedPairs[2][0]['x'], selectedPairs[2][0]['y'], 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, selectedPairs[0][0]['x'], selectedPairs[0][0]['y'], 1.0],
            [0.0, 0.0, 0.0, selectedPairs[1][0]['x'], selectedPairs[1][0]['y'], 1.0],
            [0.0, 0.0, 0.0, selectedPairs[2][0]['x'], selectedPairs[2][0]['y'], 1.0]
        ])

        vector = np.array([
            selectedPairs[0][1]['x'],
            selectedPairs[1][1]['x'],
            selectedPairs[2][1]['x'],
            selectedPairs[0][1]['y'],
            selectedPairs[1][1]['y'],
            selectedPairs[2][1]['y'],
        ])

        return self._model(matrix, vector.T)

    def _result_vector(self, vector):
        return np.array([
            [vector[0], vector[1], vector[2]],
            [vector[3], vector[4], vector[5]],
            [0.0, 0.0, 1.0]
        ])


class PerspectiveTransformation(Transformation):
    def __init__(self, heuristic):
        super().__init__(4)
        self._heuristic = heuristic

    def get_model(self, pairs):
        selectedPairs = self._heuristic.selected_pairs(pairs, limit=self.get_points_cnt())

        matrix = np.array([
            [selectedPairs[0][0]['x'], selectedPairs[0][0]['y'], 1.0, 0.0, 0.0, 0.0, -selectedPairs[0][1]['x'] * selectedPairs[0][0]['x'], -selectedPairs[0][1]['y'] * selectedPairs[0][0]['x']],
            [selectedPairs[1][0]['x'], selectedPairs[1][0]['y'], 1.0, 0.0, 0.0, 0.0, -selectedPairs[1][1]['x'] * selectedPairs[1][0]['x'], -selectedPairs[1][1]['y'] * selectedPairs[1][0]['x']],
            [selectedPairs[2][0]['x'], selectedPairs[2][0]['y'], 1.0, 0.0, 0.0, 0.0, -selectedPairs[2][1]['x'] * selectedPairs[2][0]['x'], -selectedPairs[2][1]['y'] * selectedPairs[2][0]['x']],
            [selectedPairs[3][0]['x'], selectedPairs[3][0]['y'], 1.0, 0.0, 0.0, 0.0, -selectedPairs[3][1]['x'] * selectedPairs[3][0]['x'], -selectedPairs[3][1]['y'] * selectedPairs[3][0]['x']],
            [0.0, 0.0, 0.0, selectedPairs[0][0]['x'], selectedPairs[0][0]['y'], 1.0, -selectedPairs[0][1]['y'] * selectedPairs[0][0]['x'], -selectedPairs[0][1]['y'] * selectedPairs[0][0]['y']],
            [0.0, 0.0, 0.0, selectedPairs[1][0]['x'], selectedPairs[1][0]['y'], 1.0, -selectedPairs[1][1]['y'] * selectedPairs[1][0]['x'], -selectedPairs[1][1]['y'] * selectedPairs[1][0]['y']],
            [0.0, 0.0, 0.0, selectedPairs[2][0]['x'], selectedPairs[2][0]['y'], 1.0, -selectedPairs[2][1]['y'] * selectedPairs[2][0]['x'], -selectedPairs[2][1]['y'] * selectedPairs[2][0]['y']],
            [0.0, 0.0, 0.0, selectedPairs[3][0]['x'], selectedPairs[3][0]['y'], 1.0, -selectedPairs[3][1]['y'] * selectedPairs[3][0]['x'], -selectedPairs[3][1]['y'] * selectedPairs[3][0]['y']]
        ])

        vector = np.array([
            selectedPairs[0][1]['x'],
            selectedPairs[1][1]['x'],
            selectedPairs[2][1]['x'],
            selectedPairs[3][1]['x'],
            selectedPairs[0][1]['y'],
            selectedPairs[1][1]['y'],
            selectedPairs[2][1]['y'],
            selectedPairs[3][1]['y'],
        ])

        return self._model(matrix, vector.T)

    def _result_vector(self, vector):
        return np.array([
            [vector[0], vector[1], vector[2]],
            [vector[3], vector[4], vector[5]],
            [vector[6], vector[7], 1.0]
        ])
