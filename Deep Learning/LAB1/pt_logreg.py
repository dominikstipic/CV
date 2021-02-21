import torch
from torch import nn
import numpy as np
from torch.optim import SGD
import data
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class LogReg(nn.Module):
    def __init__(self, in_features, out_features):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.sm = nn.Softmax(dim=1) 
    
    def forward(self, x):
        S = self.fc(x)
        P = self.sm(S)
        return P
    
    def loss(self, P, Yoh_):
        probs = P[Yoh_ == True]
        log_probs = torch.log(probs)
        return -torch.mean(log_probs)
    

def get_data(nclasses = 2, nsamples = 100):
    #X,Y_ = data.sample_gmm_2d(ncomponents, nclasses, nsamples)
    X,Y_ = data.sample_gauss_2d(nclasses, nsamples)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(Y_)

def logreg_eval(model, X):
    if X.dim() < 2:
        X = X.view(1,-1)
    model.eval()
    with torch.no_grad():
        P = model(X)
    return P
        

def logreg_train(X, Y_, model, optimizer, param_niter=10000, param_delta=0.05, param_lambda=10e-3, verbose=False, gradient_check=False):
    optimizer = optimizer(model.parameters(), lr = param_delta, weight_decay = param_lambda)
    X.requires_grad=True
    
    Yoh_ = torch.tensor(data.class_to_onehot(Y_))
    model.train()
    for i in range(param_niter):
        P = model(X)
        L = model.loss(P, Yoh_)
        if i % 10 == 0 and verbose:
            print(f"iter={i}, loss={L}")
        
        optimizer.zero_grad()
        L.backward()
        optimizer.step()
        
        if gradient_check and i % 10 == 0:
            with torch.no_grad():
                N = len(X)
                W, b = model.fc.weight, model.fc.bias
                scores = (W @ X.T + b.view(-1,1)).T
                P = torch.exp(scores)/torch.sum(torch.exp(scores),dim=1).view(-1,1)
                
                dL_ds = (P - Yoh_).float()
                dL_dw = 1/N*dL_ds.T @ X
                dL_db = 1/N*torch.sum(dL_ds, dim = 0)
                
                grad_W = W.grad
                grad_b = b.grad
                
                diff_w = (dL_dw - grad_W)
                diff_b = (dL_db - grad_b)
                to_print = lambda diff : "CORRECT" if ((diff < 10e-3).all()).item() else diff  
                print(f"*** w:{to_print(diff_w)}")
                print(f"*** b:{to_print(diff_b)}")
    return model


def task(model, X, Y_, hyper={"alpha" : 0.01, "delta" : 0.05, "niter" : 5000, "hidden" : 5}):
    
    niter  = hyper["niter"]
    delta  = hyper["delta"]
    niter  = hyper["niter"]
    alpha  = hyper["alpha"]
    
    optimizer = SGD
    model = logreg_train(X, Y_, model, optimizer, param_niter=niter, param_delta=delta, param_lambda=alpha, verbose=False)
    P = logreg_eval(model, X)
    
    probs, Y = torch.max(P, dim=1)
    probs, Y = probs.numpy(), Y.numpy()

    x,y_ = X.detach().numpy(), Y_.numpy()    
    bbox=(np.min(x, axis=0), np.max(x, axis=0)) 
    
    def graph_fun(X):
        X = torch.tensor(X, dtype=torch.float32)
        P = logreg_eval(model, X)
        p,y = torch.max(P,dim=1)
        return y.numpy()
    data.graph_surface(graph_fun, bbox)
    data.graph_data(x, y_, Y, special=[])
    
    C, acc, precisions, recalls  = data.eval_perf_multi(Y,Y_)
    print("----------")
    print(f"acc = {acc}")
    print(f"precission = {precisions}")
    print(f"recallas = {recalls}")
    
    
if __name__ == "__main__":
    np.random.seed(7)
    nclasses = 3
    ndim     = 2
    X,Y_ = get_data(nclasses)

    
    params={"alpha" : 0, 
            "delta" : 0.1, 
            "niter" : 10000}
    
    model = LogReg(ndim, nclasses)
    task(model, X, Y_, params)
    plt.show()
    
    params["alpha"] = 0.5
    model = LogReg(ndim, nclasses)
    task(model, X, Y_, params)
    plt.show()
        
    
    
