from functools import wraps
from time import time

def get_execution_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time()
        result = func(*args, **kwargs)
        end = time()

        print(f"[{func.__name__}()] Execution time: %.4f seconds" % (end - start))
        
        return result
    
    return wrapper


@get_execution_time
def hophop(n, m, precision=4):
    out = 0
    for i in range(n, m):
        out += i;

        if i % 10000000 == 0:
            print(out)
            out = 0
    
    if out != 0:
        print(out)