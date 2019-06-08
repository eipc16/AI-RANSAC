import json, os, imageio
import numpy as np

from models.points import Point, KeyPoint
from models.image import Image
from utils.json_utils import JSONHelper

class FileHelper:
    def __init__(self, path):
        self._path = path
        self._encoder = JSONHelper()

        if not os.path.exists(self._path):
            os.makedirs(self._path)

    def load_key_points(self, path_name):

        def _parse_keypoint(line, index):
            items = line.split(" ")
            x, y = float(items[0]), float(items[1])
            
            features = np.array(list(map(lambda x: int(x), items[5:])))

            return KeyPoint(x, y, features, index=index)


        path = f"{self._path}/{path_name}.haraff.sift"
       
        with open(path, mode='r') as f:
            content = f.readlines()[2:]
            keyPoints = list(map(lambda x: _parse_keypoint(x[1], x[0]), enumerate(content)))
            return Image(keyPoints)


    def save_as_json(self, dest_path, obj_to_save):
        path = f"{self._path}/{dest_path}"
        out_list = []
        
        if isinstance(obj_to_save, list):
            for obj in obj_to_save:
                out_list.append(self._encoder.encode(obj))
        else:
            out_list = self._encoder.encode(obj_to_save)

        json_string = json.dumps(out_list, indent=4)
     
        with open(path, mode='w') as f:
            f.write(json_string)

    def load_image(self, path_name):
        return imageio.imread(f"{self._path}/{path_name}")

    def save_image(self, path_name, image):
        imageio.imwrite(f"{self._path}/{path_name}", image)
