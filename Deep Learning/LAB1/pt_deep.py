import torch
from torch import nn
import numpy as np
import data
from torch.optim import SGD
import torch.nn.functional as F
import matplotlib.pyplot as plt
from util import split_test_valid, get_batches, evaluate


class PTDeep(nn.Module):
    def __init__(self, config, activation):
        super(PTDeep, self).__init__()
        weights = []
        biases  = [] 
        for i in range(len(config)):
            if i == len(config)-1:
                break
            else:
                w = torch.tensor(np.random.normal(0,1,(config[i+1], config[i])), dtype=torch.float32, requires_grad=True)
                b = torch.tensor(np.random.normal(0,1,config[i+1]), dtype=torch.float32, requires_grad=True)
                weights.append(nn.Parameter(w, requires_grad=True))
                biases.append(nn.Parameter(b, requires_grad=True))
        self.weights = nn.ParameterList(weights)
        self.biases = nn.ParameterList(biases) 
        self.activation = activation
        self.softmax = F.softmax

    def forward(self, X):
        depth = len(self.weights)
        fun = self.activation
        for i in range(depth):
            W,b = self.weights[i], self.biases[i]
            X = (W @ X.T + b.view(-1,1)).T
            if i != depth-1:
                X = fun(X)
        return X
    
    def loss(self, S, Yoh, mean_loss=True):
        delta = 10e-6
        log_P = torch.log(self.softmax(S, dim=1)+delta)
        log_probs = torch.sum(Yoh*log_P, dim=1)
        L = []
        if mean_loss:
            L = -torch.mean(log_probs)
        else:
            L = -torch.sum(log_probs)
        return L
        
def deep_train(X, Y_, model, optimizer, param_niter=10000, param_delta=0.01, param_lambda=10e-3, verbose=False):
    Yoh_ = torch.tensor(data.class_to_onehot(Y_), dtype=torch.float32)
    X,Y_ = torch.tensor(X, dtype=torch.float32), torch.tensor(Y_)
    optimizer = optimizer(model.parameters(), lr = param_delta, weight_decay = param_lambda)
    model.train()
    for i in range(param_niter):
        S = model(X)
        L = model.loss(S, Yoh_)
        if i % 10 == 0 and verbose:
            print(f"iter={i}, loss={L}")
        
        optimizer.zero_grad()
        L.backward()
        optimizer.step()
        
    return model

def deep_eval(model, X):
    if type(X) != torch.Tensor:
        X = torch.tensor(X, dtype = torch.float32, requires_grad=True)
    if X.dim() < 2:
        X = X.view(1,-1)
    model.eval()
    with torch.no_grad():
        S = model(X)
    return S
 
def deep_train_mb(X, Y_, model, batch_size=10, param_niter=10000, param_delta=0.01, param_lambda=10e-3, verbose=False):
    optimizer = SGD(model.parameters(), lr=param_delta, weight_decay=param_lambda)
    
    x_train, y_train, x_valid, y_valid = split_test_valid(X, Y_) 
    Yoh_train = torch.tensor(data.class_to_onehot(y_train))
    Yoh_valid = torch.tensor(data.class_to_onehot(y_valid))
    
    N = len(x_train)
    train_error = []
    valid_error = []
    for i in range(param_niter):
        np.random.shuffle(x_train)
        x_train_batches, y_train_batches = get_batches(x_train, Yoh_train, batch_size) 
        model.train()
        
        train_loss = 0
        valid_loss = 0
        for x_train_batch, y_train_batch in zip(x_train_batches, y_train_batches):
            S = model(x_train_batch)
            L = model.loss(S, y_train_batch, mean_loss=False)
            train_loss += L
            optimizer.zero_grad()
            L.backward()
            optimizer.step()
        train_loss /= N
        with torch.no_grad():
            S = deep_eval(model, x_valid)
            valid_loss = model.loss(S, Yoh_valid)
            
        if i % 10 == 0:
            train_error.append(train_loss)
            valid_error.append(valid_loss)
            print(f"iter={i}, train_loss={train_loss/N}, valid_loss={valid_loss/N}")
    return model,train_error,valid_error

#######################################3

def count_params(model):
    acc = 0
    for name,param in model.named_parameters():
        dim = np.array(param.shape)
        acc += np.product(dim)
        print(f"*** {name} : {dim}")
    print(f"Total:{acc}")


def routine(model, X, Y_, hyper={"alpha" : 0.01, "delta" : 0.05, "niter" : 5000}):
    niter  = hyper["niter"]
    delta  = hyper["delta"]
    niter  = hyper["niter"]
    alpha  = hyper["alpha"]
    
    m = deep_train(X, Y_, model, SGD, param_niter=niter, param_delta=delta, param_lambda=alpha, verbose=False)
    P = deep_eval(m, X).numpy()
    evaluate(m, X, Y_, P)
    graph(m, X, Y_)

def graph(model, X, Y_):
    P = deep_eval(model, X)
    P = P.numpy()
    Y = np.argmax(P,axis=1)
    bbox=(np.min(X, axis=0), np.max(X, axis=0)) 
    def graph_fun(X):
        X = torch.tensor(X, dtype=torch.float32)
        P = deep_eval(model, X)
        P = P.numpy()
        return P[:,1]
    data.graph_surface(graph_fun, bbox)
    data.graph_data(X, Y_, Y, special=[])
    plt.show()
    

def task1():
    np.random.seed(100)
    X,Y_ = data.sample_gauss_2d(C=2, N=50)
    model = PTDeep([2,10,10,2], F.relu)
    model = deep_train(X, Y_, model, SGD, param_niter=10000, param_delta=0.1, param_lambda=0, verbose=False)
    graph(model, X, Y_)

def task2():
    model = PTDeep([2,10,10,2], F.relu)
    count_params(model)

def task3():
    np.random.seed(100)
    X1, Y1_ = data.sample_gmm_2d(ncomponents=4, nclasses=2, nsamples=40)
    X2, Y2_ = data.sample_gmm_2d(ncomponents=6, nclasses=2, nsamples=10)
    model1 = PTDeep([2,2], F.relu)
    model2 = PTDeep([2,10,2], F.relu)
    model3 = PTDeep([2,10,10,2], F.relu)
    
    
    params={"alpha" : 0, 
            "delta" : 0.1, 
            "niter" : 3000}
    
    routine(model1, X1, Y1_, params)
    routine(model2, X1, Y1_, params)
    routine(model3, X1, Y1_, params)
    
    routine(model1, X2, Y2_, params)
    routine(model2, X2, Y2_, params)
    routine(model3, X2, Y2_, params)
    
def task4():
    np.random.seed(100)
    X, Y_ = data.sample_gmm_2d(ncomponents=4, nclasses=2, nsamples=40)
    model = PTDeep([2,10,2], F.relu)
    
    params={"alpha" : 0, 
            "delta" : 0.1, 
            "niter" : 3000}
    routine(model, X, Y_, params)
    
    model = PTDeep([2,10,2], F.softmax)
    routine(model, X, Y_, params)
    
    



if __name__ == "__main__":
    np.random.seed(100)
    

    #task1()
    #task2()
    #task3()
    task4()
    
    