import numpy as np
import data
import matplotlib.pyplot as plt

def gradchecker(grad, fun, W, b):
    approx_w = []
    approx_b = []
    EPSILON = 10**-3
    E = np.eye(W.shape[1])
    
    for i in range(W.shape[0]):
        for e in E:
            delta = np.zeros_like(W)
            delta[i] = W[i]+e*EPSILON
            delta[i, e == 0] = 0 
            print(delta)
            approx = (fun(W+delta, b) - fun(W-delta, b))/(2*EPSILON)
            approx_w.append(approx)
    
    for e in E:
        delta = e*EPSILON
        approx = (fun(b+delta, b) - fun(b-delta, b))/(2*EPSILON)
        approx_b.append(approx)
    
    grad_W,grad_b = grad
    
    print("real_w: {}  approx: {} ".format(grad_W, approx_w))
    print("real_b: {}  approx: {} ".format(grad_b, approx_b))
    pass

def softmax(x):
    exp_x_shifted = np.exp(x - np.max(x))
    sums = np.sum(exp_x_shifted, axis=1).reshape(-1,1)
    sums = np.tile(sums, x.shape[1])
    return exp_x_shifted/sums


def one_hot_encode(Y):
    Yoh = np.zeros((len(Y), max(Y)+1))
    Yoh[range(len(Y)), Y] = 1
    return Yoh

def logreg_train(X, Y_, param_niter=10000, param_delta=0.1, gradcheck=False, verbose=False):
    N,D = X.shape
    C   = max(Y_)+1

    W = np.random.normal(loc=0, scale=1, size=(C,D))
    b = np.random.normal(loc=0, scale=1, size=(C,1))
    
    Yoh = one_hot_encode(Y_)
    for i in range(param_niter):
        scores = (W @ X.T + np.tile(b, N)).T
        probs  = softmax(scores)
        
        probs_true = probs[Yoh == 1]
        loss   = -sum((np.log(probs_true)))
        if i % 10 == 0 and verbose:
            print("iteration {}: loss {}".format(i, loss))
            
        
        dL_ds = probs - Yoh
        
        grad_W = 1/N*(dL_ds.T @ X)
        grad_b = 1/N*sum(dL_ds).reshape(-1,1)
        
        if gradcheck:
            def fun(W,b):
                scores = (W @ X.T + np.tile(b, N)).T
                probs  = softmax(scores)
                
                probs_true = probs[Yoh == 1]
                loss   = -sum((np.log(probs_true)))
                return loss
            gradchecker((grad_W,grad_b), fun, W, b)
        
        W += -param_delta * grad_W
        b += -param_delta * grad_b
    return W, b.flatten()

def logreg_classify(X, W, b):
    b = b.reshape(-1,1)
    B = np.tile(b, X.shape[0])
    scores = (W @ X.T + B).T
    probs  = softmax(scores)
    return probs

def eval_perf_multi(Y, Y_):
    C = np.zeros((max(Y_)+1,max(Y_)+1))
    for i in range(len(Y)):
        y_pred = Y[i]
        y      = Y_[i]
        C[y_pred,y] += 1
    
    matrices = []
    classes = max(Y_)+1
    for i in range(classes):
        C_i = np.zeros((2,2))
        C_i[0,0] = C[i,i]
    
        horizontal = C[i, :]
        C_i[0,1] = np.delete(horizontal,i).sum()
        
        vertical = C[:,i]
        C_i[1,0] = np.delete(vertical,i).sum()
        
        A = np.delete(C, i, 0)
        A = np.delete(A, i, 1)    
        C_i[1,1] = A.sum()
        matrices.append(C_i)
        
    precisions = []
    recalls    = []

    for C_i in matrices:
        p = C_i[0,0]/(C_i[0,0]+C_i[0,1])
        r = C_i[0,0]/(C_i[0,0]+C_i[1,0])
        precisions.append(p)
        recalls.append(r)
        
    acc = (C[0,0] + C[1,1] + C[2,2])/C.sum()
    return C, acc, np.array(precisions), np.array(recalls)
        
     
if __name__=="__main__":
    np.random.seed(100)
    
    X,Y_ = data.sample_gauss_2d(3,100)
    W,b  = logreg_train(X, Y_, gradcheck=False, verbose=True) 
    
    probs = logreg_classify(X, W, b)
    Y = np.apply_along_axis(np.argmax, 1, probs)
    
    _, acc, precisions, recalls =  eval_perf_multi(Y, Y_)
    
    print("acc:{}, precissions:{}, recalls:{}".format(acc, precisions, recalls))
    
    bbox=(np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(lambda X : np.apply_along_axis(np.argmax, 1, logreg_classify(X, W, b)), bbox, offset=0.5)
    data.graph_data(X, Y_, Y, special=[])
    plt.show()
    
    
    
    
    
    
    
    
    
    


