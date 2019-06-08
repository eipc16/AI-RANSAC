import numpy as np
from functools import partial
from models.points import KeyPoint
from multiprocessing import Pool
from utils.time_utils import get_execution_time


class Image:
    def __init__(self, keypoints):
        self._keypoints = keypoints

    def get_keypoints_pairs(self, picture):
        # return list(map(lambda ))
        pass

    @get_execution_time
    def nearest_keypoints_indexes(self, picture):
        pool = Pool()
        copier = partial(self.nearest_keypoint_index, keypoint_list=picture._keypoints)
        return pool.map(copier, self._keypoints)

    @staticmethod
    def nearest_keypoint_index(keypoint, keypoint_list):
        distances = list(map(lambda x: keypoint.feature_dist(x), keypoint_list))
        return distances.index(min(distances))