import json, os, imageio
import numpy as np

from models.points import Point, KeyPoint
from json_utils import JSONHelper


class FileHelper:
    def __init__(self, path):
        self._path = path
        self._encoder = JSONHelper()

        if not os.path.exists(self._path):
            os.makedirs(self._path)

    def load_key_points(self, path_name):

        def _parse_keypoint(line):
            items = line.split(" ")
            x, y = float(items[0]), float(items[1])
            
            features = map(items[5:], lambda x: int(x))

            return KeyPoint(x, y, features)


        path = f"{self._path}/{path_name}"
       
        with open(path, mode='r') as f:
            content = f.readlines()[2:]
            keyPoints = map(content, lambda x : _parse_keypoint(x))



    def save_as_json(self, dest_path, obj_list):
        path = f"{self._path}/{dest_path}"
        out_list = []
        
        for obj in obj_list:
            out_list.append(self._encoder.encode(obj))

        json_string = json.dumps(out_list, indent=4)
     
        with open(path, mode='w') as f:
            f.write(json_string)

    def load_image(self, path_name):
        return imageio.imread(f"{self._path}/{path_name}")

    def save_image(self, path_name, image):
        imageio.imwrite(f"{self._path}/{path_name}", image)

