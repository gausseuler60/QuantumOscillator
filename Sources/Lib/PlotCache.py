import os
import pickle
from sys import argv

cache_dir = os.path.join(os.getcwd(), 'Cache')
 
def IsInCache(filename):
    if len(argv)>1:
        if argv[1]=='-u':
            return False
    return os.path.isfile(os.path.join(cache_dir, filename))

def SaveSolution(array, filename):
    
    if not os.path.isdir(cache_dir):
        os.mkdir(cache_dir)
    with open(os.path.join(cache_dir, filename), 'wb') as f:
        pickle.dump(array, f)

def ReadSolution(filename):
    with open(os.path.join(cache_dir, filename), 'rb') as f:
        data = pickle.load(f)
        return data
