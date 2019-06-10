import json, os
import numpy as np

from models.points import KeyPoint
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
        
        def _recursive_list_encoder(target_list):
            if not (isinstance(target_list, list) or isinstance(target_list, np.ndarray)):
                return self._encoder.encode(target_list)

            out_obj = []

            for obj in target_list:
                if isinstance(obj, list) or isinstance(obj, np.ndarray):
                    out_obj.append(_recursive_list_encoder(obj))
                else:
                    out_obj.append(self._encoder.encode(obj))

            return out_obj

        out_list = _recursive_list_encoder(obj_to_save)
        json_string = json.dumps(out_list, indent=4)
     
        with open(path, mode='w') as f:
            f.write(json_string)

    def load_from_json(self, file_path):
        path = f"{self._path}/{file_path}"

        def parse_obj(target):
            if isinstance(target, list):
                output = []
                for i in target:
                    output.append(parse_obj(i))
                return output
            elif isinstance(target, str):
                return json.loads(target)
            else:
                return target

        with open(path, mode='rb') as f:
            output = json.load(f)
            output = parse_obj(output)

        return output