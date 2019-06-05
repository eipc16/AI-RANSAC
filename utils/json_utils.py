from json import JSONEncoder

class JSONHelper(JSONEncoder):
    def default(self, o):
        return o.__dict__