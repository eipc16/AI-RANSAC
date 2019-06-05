import numpy as np

class Image:
    def __init__(self, keypoints):
        self._keypoints = keypoints

    def get_keypoints_pairs(self, picture):

    def nearest_keypoint_index(keypoint, keypoint_list):
        distances = list(map(lambda x: distance(keypoint, x), keypoint_list))
        return keypoint_list.index(min(distances))

    def distance(point1, point2):
        feature_pairs = zip(point1.feautres, point2.features)
        return np.sqrt(reduce(lambda x: np.square(x[0] - x[1]), feature_pairs))