from json import JSONEncoder
import numpy as np

class JSONHelper(JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray):
            if o.flags['C_CONTIGUOUS']:
                obj_data = o.data
            else:
                cont_obj = np.ascontiguousarray(o)
                assert(cont_obj.flags['C_CONTIGUOUS'])
                obj_data = cont_obj.data

            return o.tolist()

        try:
            obj_dict = o.__dict__
        except TypeError:
            pass
        else:
            return obj_dict
            
        return JSONEncoder.default(self, o)