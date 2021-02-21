from sklearn.svm import SVC
import matplotlib.pyplot as plt
import data
import numpy as np
from pt_deep import PTDeep, deep_train
from torch.optim import SGD
import torch
import torch.nn.functional as F

class KSVMWrap():
    def __init__(self, X, Y_, param_svm_c=1, param_svm_gamma="auto"):
        model = SVC(C=param_svm_c, gamma=param_svm_gamma, probability=True)
        self.model = model.fit(X,Y_)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def get_scores(self, X):
        return self.model.predict_proba(X)
    
    def support(self):
        return self.model.support_vectors_
    
    def support_index(self):
        return self.model.support_
        
    
def graph(model, X, Y_):
    bbox=(np.min(X, axis=0), np.max(X, axis=0)) 
    Y = model.predict(X)
    data.graph_surface(lambda X : model.get_scores(X)[:,1], bbox)
    data.graph_data(X, Y_, Y, special=model.support_index())
    plt.show()    
    

def evaluate(model, X, Y_):
    Y = model.predict(X)
    P = model.get_scores(X)
    probs = np.max(P,axis=1)
    C, acc, precisions, recalls  = data.eval_perf_multi(Y,Y_)
    Yr = Y[np.argsort(probs)]
    ap = data.eval_AP(Yr)
    print("----------")
    print(f"acc = {acc}")
    print(f"precission = {precisions}")
    print(f"recalls = {recalls}")
    print(f"ap = {ap}")
    print("----------") 
    
def get_ap(fun, X, Y_):
    P = fun(X)
    probs = np.max(P,axis=1)
    Y     = np.argmax(P, axis=1)
    Yr = Y[np.argsort(probs)]
    ap = data.eval_AP(Yr)
    return ap
    
def compare(svm, nn, niter=20):
    ap1 = []
    ap2 = []
    for i in range(niter):
        X,Y_ = data.sample_gmm_2d(ncomponents=3, nclasses=2, nsamples=100)
        if Y_.sum() == 0 or Y_.sum() == 300:
            continue
        
        s = svm(X, Y_, param_svm_c=1)
        nn  = deep_train(X, Y_, nn, SGD, param_niter=1000)
        
        ap_svm = get_ap(s.get_scores, X, Y_)
        ap_nn  = get_ap(lambda x : nn.forward(torch.tensor(x, dtype=torch.float32)).detach().numpy(), X, Y_)
        print(f"iter={i}, ap_svm={ap_svm}, ap_nn={ap_nn}")
        ap1.append(ap_svm)
        ap2.append(ap_nn)
    ap1 = np.array(ap1)
    ap2 = np.array(ap2)
    return ap1.mean(), ap2.mean(), ap1.std(), ap2.std()


def task1():
    np.random.seed(100)
    X,Y_ = data.sample_gmm_2d(ncomponents=3, nclasses=2, nsamples=100)
    model = KSVMWrap(X, Y_, param_svm_c=1)
    graph(model, X, Y_)
    evaluate(model, X, Y_)

def task2():
    svm = KSVMWrap
    nn  = PTDeep([2, 5, 2], F.relu)
    
    ap1, ap2, s1, s2 = compare(svm,nn, niter=5)
    print(f"mean : ap(svm)={ap1}, ap(nn)={ap2}")
    print(f"std: ap(svm)={s1}, ap(nn)={s2}")
    
    

def sigmoid(X):
    X[:,1] = 1/(1+torch.exp(-X[:,1]))
    X[:,0] = 1 - X[:,1]
    return X

if __name__ == "__main__":
    task1()
    #task2()
# =============================================================================
#     svm = KSVMWrap
#     nn  = PTDeep([2, 5, 2], sigmoid)
#     
#     ap1, ap2, s1, s2 = compare(svm,nn)
#     print(ap1, ap2)
#     print(s1, s2)
# =============================================================================
