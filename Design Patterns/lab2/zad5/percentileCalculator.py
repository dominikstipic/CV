import math
import numpy as np

def nearest_rank_method(xs, p):
    xs = sorted(xs)
    N = len(xs)
    indexes = np.arange(1,N+1)
    n_p = p*N/100 + 0.5

    m1 = zip(map(lambda x : abs(x-n_p),indexes),indexes)
    i = min(m1,key=lambda x:x[0])[1] - 1
    return xs[i] 

def interpolation_method(xs,p):
     xs = sorted(xs) 
