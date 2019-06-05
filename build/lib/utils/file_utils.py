import json, os
import numpy as np
from ransacai.models.points import Point

class FileHelper:
    def __init__(self, path):
        self._path = path

        if not os.path.exists(self._path):
            os.makedirs(self._path)

    def save_as_json(self, dest_path, obj_list):
        path = f"{self._path}/{dest_path}"
        fixed_list = obj_list.tolist()
        json_string = json.dumps(fixed_list)
        
        with open(path, mode='w') as f:
            f.write(json_string)

x = Point(3, 4)
y = Point(2, 4)
z = Point(1, 2)

x_arr = [x, y, z]
print(x)
helper = FileHelper('../files')
helper.save_as_json('file.json', x_arr)