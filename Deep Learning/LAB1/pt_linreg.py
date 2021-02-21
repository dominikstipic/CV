import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions.normal import Normal
from torch.optim import SGD


class LinearRegression(nn.Module):
    def __init__(self, in_features, out_features):
        super(LinearRegression, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
    
    def forward(self, x):
        return self.fc(x)
    
def create_data(N, noise, generating_fun):
    fun = lambda x : 5 + x - 2*x**2 - 5*x**3
    X = np.linspace(-5,5,N)
    Y = fun(X) + np.random.normal(0, noise, size=len(X))
    return torch.tensor(X).view(-1,1).float(),torch.tensor(Y).view(-1,1).float()

def linreg_train1(X, Y_, param_niter=10000, param_delta=0.05, verbose=False, gradient_check=False):
    w,b = Normal(0,1).sample(), Normal(0,1).sample()
    N = len(X)
    
    X.requires_grad=True
    w.requires_grad=True
    b.requires_grad=True
    for i in range(param_niter):
        Y = w*X + b
        loss = torch.sum((Y-Y_)**2)/N
        if i % 10 == 0 and verbose:
            print(f"iter={i}, loss={loss}")
        loss.backward()
        
        if gradient_check and i % 10 == 0:
            N = len(Y_)
            h = (w*X+b - Y_)
            dL_dw = 2/N * torch.sum(h*X)
            dL_db = 2/N * torch.sum(h)
            grad_w = w.grad
            grad_b = b.grad
            
            diff_w = (dL_dw - grad_w).item()
            diff_b = (dL_db - grad_b).item()
            to_print = lambda diff : "CORRECT" if diff < 10e-3 else diff  
            print(f"*** w:{to_print(diff_w)}")
            print(f"*** b:{to_print(diff_b)}")
        
        with torch.no_grad():
            w -= w.grad*param_delta
            b -= b.grad*param_delta
            w.grad.zero_()
            b.grad.zero_()
    return w.item(), b.item()

def linreg_train2(X, Y_, model, optimizer, loss, param_niter=10000, param_delta=0.05, verbose=False, gradient_check=False):
    optimizer = optimizer(model.parameters(), lr = param_delta)
    X.requires_grad=True
    for i in range(param_niter):
        Y = model(X)
        L = loss(Y, Y_) 
        if i % 10 == 0 and verbose:
            print(f"iter={i}, loss={L}")
        
        # Before the backward pass, use the optimizer object to zero all of the
        # gradients for the variables it will update (which are the learnable
        # weights of the model). This is because by default, gradients are
        # accumulated in buffers( i.e, not overwritten) whenever .backward() is called. 
        optimizer.zero_grad()
        L.backward()
        optimizer.step()
        
        if gradient_check and i % 10 == 0:
            w, b = model.fc.weight, model.fc.bias
            N = len(Y_)
            h = (w*X+b - Y_)
            dL_dw = 2/N * torch.sum(h*X)
            dL_db = 2/N * torch.sum(h)
            grad_w = w.grad
            grad_b = b.grad
            
            diff_w = (dL_dw - grad_w).item()
            diff_b = (dL_db - grad_b).item()
            to_print = lambda diff : "CORRECT" if diff < 10e-3 else diff  
            print(f"*** w:{to_print(diff_w)}")
            print(f"*** b:{to_print(diff_b)}")
            
    return model.fc.weight.item(), model.fc.bias.item()

if __name__ == "__main__":
    np.random.seed(100)
    choice = 1
    
    fun = lambda x : 5 + x - 2*x**2 - 5*x**3
    X,Y_ = create_data(50, noise = 200, generating_fun = fun)
    
    if choice == 1:
        w,b = linreg_train1(X, Y_, verbose=True, gradient_check=True)
    elif choice == 2:
        model = LinearRegression(1, 1)
        optimizer = SGD
        loss = lambda Y,Y_ : torch.sum((Y-Y_)**2)/len(Y)
        w,b = linreg_train2(X, Y_, model, optimizer, loss, verbose=True, gradient_check=False)
    X,Y_ = X.detach().numpy(), Y_.detach().numpy()

    plt.scatter(X, Y_)
    plt.plot(np.linspace(-5,5,100), fun(np.linspace(-5,5,100)), "r--", label = "underline function")
    plt.plot(np.linspace(-5,5,100),np.linspace(-5,5,100)*w+b, "r", label = "fit")
    plt.legend()
    plt.show()
    
    
    
    