def generate_sequence(first,last,step):
    xs = [i for i in range(first,last+step,step)]
    return xs

def sample_normal(mean,std,N):
    import numpy as np
    xs = np.random.normal(loc=mean,scale=std,size=N).tolist()
    return xs

def __fibbonaci(N):
    if N <= 0:
        return 0
    if N == 1:
        return 1
    return __fibbonaci(N-1) + __fibbonaci(N-2)

def generate_fibbonaci(N):
    assert N > 0, "number of elems can not be less than 0" 
    xs = [__fibbonaci(i) for i in range(N)]
    return xs