import torch
import torchvision
import matplotlib.pyplot as plt
from torch_deep import PTDeep,deep_eval,deep_train_mb
from torch.optim import SGD
import data
import torch.nn.functional as F
import copy
from util import evaluate, split_test_valid

def get_data():
    dataset_root = '/tmp/mnist'  
    mnist_train = torchvision.datasets.MNIST(dataset_root, train=True, download=True)
    mnist_test = torchvision.datasets.MNIST(dataset_root, train=False, download=True)

    x_train, y_train = mnist_train.train_data, mnist_train.train_labels
    x_test, y_test = mnist_test.test_data, mnist_test.test_labels
    x_train, x_test = x_train.float().div_(255.0), x_test.float().div_(255.0)
    return x_train, y_train, x_test, y_test


def preprocess(x_train, y_train, x_test, y_test):
    x_train = x_train.view(len(x_train), 28*28)
    x_test = x_test.view(len(x_test), 28*28) 
    return x_train, y_train, x_test, y_test

def show_weights(model):
    params = []
    for param in model.named_parameters():
        params.append(param)
        print(param)
    
def deep_train(X, Y_, model, optimizer, param_niter=10000, param_delta=0.01, param_lambda=10e-3, verbose=False):
    if type(X) != torch.Tensor:
        X = torch.tensor(X, dtype = torch.float32, requires_grad=True)
    optimizer = optimizer(model.parameters(), lr = param_delta, weight_decay = param_lambda)
    Yoh_ = torch.tensor(data.class_to_onehot(Y_))
    model.train()
    errors = []
    weights = []
    for i in range(param_niter):
        S = model(X)
        L = model.loss(S, Yoh_)
        errors.append(L.detach().numpy())
        weights.append(model.weights)
        if i % 10 == 0 and verbose:
            print(f"iter={i}, loss={L}")
        optimizer.zero_grad()
        L.backward()
        optimizer.step()
    return model, errors, weights    
 
def task1(x_train, y_train, x_test, y_test):
    """
        Naučene težine predstavljaju obrazac(template) po kojem model raspoznaje slike.
    """
    x_train, y_train, x_test, y_test = get_data()
    x_train, y_train, x_test, y_test = preprocess(x_train, y_train, x_test, y_test)
    
    model = PTDeep([784, 10], [])
    model,_,_ = deep_train(x_train, 
               y_train, 
               model, 
               SGD, 
               param_niter=10000, 
               param_delta=0.01, 
               param_lambda=10e-3, 
               verbose=True)
    xs = []
    for param in model.named_parameters():
        xs.append(param[1])
    W = xs[0]
    
    for W_i in W:
        W_i = W_i.detach().view(28,28).numpy()
        plt.imshow(W_i, plt.get_cmap('gray'))
        plt.show()

def task2(x_train, y_train, x_test, y_test):
    model1 = PTDeep([784, 10], F.relu)
    model2 = PTDeep([784, 100,10], F.relu)
    
    model1,errors1,weights1 = deep_train(
                               x_train, 
                               y_train, 
                               model1, 
                               SGD, 
                               param_niter=10000, 
                               param_delta=0.01, 
                               param_lambda=10e-3, 
                               verbose=True)
    
    model2,errors2,weights2 = deep_train(
                               x_train, 
                               y_train, 
                               model2, 
                               SGD, 
                               param_niter=10000, 
                               param_delta=0.01, 
                               param_lambda=10e-3, 
                               verbose=True)
    
    plt.plot(errors1,label = "model1")
    plt.plot(errors2,label = "model2")
    evaluate(model1, x_train, y_train)
    evaluate(model2, x_train, y_train)
    plt.legend()
    plt.show()
    
    
def task3(x_train, y_train, x_test, y_test):
    x_train, y_train, x_valid, y_valid = split_test_valid(x_train, y_train) 
    
    model = PTDeep([784, 10], F.relu)
    param_niter=1000 
    param_delta=0.1
    param_lambda=10e-3 
    optimizer = SGD(model.parameters(), lr=param_delta, weight_decay=param_lambda)
    
    Yoh_train = torch.tensor(data.class_to_onehot(y_train))
    Yoh_valid = torch.tensor(data.class_to_onehot(y_valid))
    train_errors = []
    valid_errors = []
    
    best_parameters = model.state_dict()
    best_loss = []
    for i in range(param_niter):
        model.train()
        S = model(x_train)
        L = model.loss(S, Yoh_train)
        train_errors.append(L)
        if i % 10:
            print(f"iter={i}, loss={L}")
        
        optimizer.zero_grad()
        L.backward()
        optimizer.step()
        
        with torch.no_grad():
            S = deep_eval(model, x_valid)
            loss = model.loss(S, Yoh_valid)
            if not best_loss:
                best_loss = loss
            valid_errors.append(loss)
            if loss < best_loss:
                best_parameters = copy.deepcopy(model.state_dict())
    
    plt.plot(train_errors, label = "train")
    plt.plot(valid_errors, label = "valid")
    
    model = model.load_state_dict(best_parameters)  
    evaluate(model, x_test, y_test)
    return model
       
def task4(x_train, y_train, x_test, y_test):
    model = PTDeep([784, 10], F.relu)
    model,te,ve = deep_train_mb(x_train, y_train, model, param_niter=1000, verbose=True)
    return model,te,ve
    

if __name__ == "__main__":
    x_train, y_train, x_test, y_test = get_data()
    d = preprocess(x_train, y_train, x_test, y_test)
    x_train, y_train, x_test, y_test = d
    
    #task2(*d)
    #model = task3(*d)
    model,te,ve = task4(*d)
    plt.plot(te, label = "train")
    plt.plot(ve, label = "valid")
    plt.legend()
    plt.show()
    