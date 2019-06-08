import numpy as np
from functools import partial
from itertools import compress
from models.points import KeyPoint
from multiprocessing import Pool
from utils.time_utils import get_execution_time
import tqdm


class Image:
    def __init__(self, keypoints):
        self._keypoints = keypoints if isinstance(keypoints, np.ndarray) else np.array(keypoints)

    @get_execution_time
    def get_keypoint_pairs(self, picture):
        return [(self._keypoints[i], picture._keypoints[j]) for i, j in self.closest_indexes_pairs(picture)]

    @get_execution_time
    def closest_indexes_pairs(self, picture):
        def _val_or_null(array, index):
            if index >= array.shape[0]:
                return None
            return array[index]

        left = self.nearest_keypoints_indexes(picture)
        right = picture.nearest_keypoints_indexes(self)
        
        correct_pairs = np.array([i == _val_or_null(right, left[i]) for i in range(left.shape[0])])
        left_points = np.array(list(compress(range(correct_pairs.shape[0]), correct_pairs)))

        zipped_points = zip(left_points, left[left_points])

        return set(zipped_points)
    
    @get_execution_time
    def nearest_keypoints_indexes(self, target):
        pool = Pool()
        copier = partial(self._nearest_keypoint_index, keypoint_list=target._keypoints)
        return np.array(pool.map(copier, self._keypoints))

    @staticmethod
    def _nearest_keypoint_index(keypoint, keypoint_list):
        distances = np.array([keypoint.feature_dist(x) for x in keypoint_list])
        return np.argmin(distances)

    @staticmethod
    def get_pairs_dict(keypoint_pairs):
        return list(map(lambda x: (x[0].toJSON(), x[1].toJSON()), keypoint_pairs))