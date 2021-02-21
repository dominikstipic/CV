import matplotlib.pyplot as plt
from data import sample_gmm_2d, class_to_onehot,graph_surface,graph_data
import numpy as np
import data

def scatter_data(X, Y_):
    X_plus = X[Y_ == 1]
    X_neg  = X[Y_ == 0]
    plt.scatter(X_plus[:,0], X_plus[:, 1])
    plt.scatter(X_neg[:,0], X_neg[:, 1])


def relu(S):
    X = S.copy()
    X[X < 0] = 0
    return X

def softmax(S):
    _, D = S.shape
    exp_shifted = np.exp(S-np.max(S))
    sums = np.sum(exp_shifted, axis = 1).reshape(-1,1)
    sums = np.tile(sums, D)
    return exp_shifted/sums

def reshape_like(X, template):
    X = X.copy()
    X = X.reshape(*template.shape)
    return X
    
def gradient_checking(loss_fun, params, grads, epsilon=10**-3):
    grad_W1, grad_W2, grad_b1, grad_b2 = grads
    params_flatten = list(map(lambda w : w.flatten(), params))
    
    approx = []
    for i, theta in enumerate(params_flatten):
        E = np.eye(len(theta))
        grad_theta = []
        for e in E:
            delta = e*epsilon
            theta_plus = theta + delta
            theta_minus = theta - delta
            params_plus = params_flatten.copy()
            params_plus[i] = theta_plus 
            params_minus = params_flatten.copy()
            params_minus[i] = theta_minus
            
            params_plus = [reshape_like(params_plus[i], params[i]) for i in range(len(params))]
            params_minus = [reshape_like(params_minus[i], params[i]) for i in range(len(params))]
            approx_grad  =  (loss_fun(*params_plus) - loss_fun(*params_minus)) / (2*epsilon)
            
            grad_theta.append(approx_grad)
        approx.append(np.array(grad_theta))
    
    approx_W1, approx_W2, approx_b1, approx_b2 = approx
    
    grad_W1 = grad_W1.flatten()
    grad_W2 = grad_W2.flatten()
    
    print_fun = lambda diff : "CORRECT" if (np.abs(diff) < 0.01).all() else diff
    print(f"*** grad_W1-approx_W1: {print_fun(grad_W1 - approx_W1)}")
    print(f"*** grad_W2-approx_W2: {print_fun(grad_W2 - approx_W2)}")
    print(f"*** grad_b1-approx_b1: {print_fun(grad_b1 - approx_b1)}")
    print(f"*** grad_b2-approx_b2: {print_fun(grad_b2 - approx_b2)}")
            
    
    
def fcann2_train(X, Y_, param_niter=10000, param_delta=0.05, param_lambda=1e-3, hidden=5, verbose=False, grad_check=False):
    C = max(Y_) + 1
    N,D = X.shape
    H = hidden
    
    W1 = np.random.normal(0, 1, (H,D))  # 3 x 2
    b1 = np.random.normal(0, 1, H).reshape(-1,1)  # 3 x 1
    W2 = np.random.normal(0, 1, (C,H))  # 2 X 3
    b2 = np.random.normal(0, 1, C).reshape(-1,1) # 2 x 1
    
    Yoh = class_to_onehot(Y_)
    for i in range(param_niter):
        S1 = W1@X.T + np.tile(b1, N)
        S1 = S1.T
        
        H1 = relu(S1)
        S2 = W2@H1.T + np.tile(b2, N)
        S2 = S2.T
        P  = softmax(S2)
        
        probs = P[Yoh == 1]
        loss  = -1/N*np.log(probs).sum()
        
        if i % 10 == 0 and verbose:
            print(f"iter={i}, loss={loss}")
        
        Gs2  = P - Yoh
        grad_W2 = 1/N*Gs2.T @ H1
        grad_b2 = 1/N*Gs2.sum(axis=0)
        
        Gh1     = Gs2 @ W2
        dH1_ds1 = (np.eye(H) * np.diag(S1) > 0)*1
        Gs1     = Gh1@dH1_ds1
        grad_W1 = 1/N*Gs1.T @ X
        grad_b1 = 1/N*Gs1.sum(axis=0)
        
        if i % 10 == 0 and grad_check:
            def loss_fun(W1, W2, b1, b2):
                S1 = W1@X.T + np.tile(b1, N)
                S1 = S1.T
                H1 = relu(S1)
                S2 = W2@H1.T + np.tile(b2, N)
                S2 = S2.T
                P  = softmax(S2)
                probs = P[Yoh == 1]
                loss  = -1/N*np.log(probs).sum()
                return loss
            
            params = (W1,W2,b1,b2)
            grads  = (grad_W1, grad_W2, grad_b1, grad_b2)
            gradient_checking(loss_fun, params, grads)
        
        W1 = W1*(1-param_delta*param_lambda) - param_delta*grad_W1
        W2 = W2*(1-param_delta*param_lambda) - param_delta*grad_W2
        b1 -= (param_delta*grad_b1).reshape(-1,1)
        b2 -= (param_delta*grad_b2).reshape(-1,1)
        
    return W1, W2, b1.flatten(), b2.flatten()
        
def fcann2_classify(X, W1, W2, b1, b2):
    b1,b2 = b1.reshape(-1,1), b2.reshape(-1,1)
    N = len(X)
    
    S1 = W1@X.T + np.tile(b1, N)
    S1 = S1.T
    H1 = relu(S1)
    S2 = W2@H1.T + np.tile(b2, N)
    S2 = S2.T
    P  = softmax(S2)
    return P, P.argmax(axis=1)

def task(X, Y_, hyper={"alpha" : 0.01, "delta" : 0.05, "niter" : 5000, "hidden" : 5}):
    
    niter  = hyper["niter"]
    delta  = hyper["delta"]
    niter  = hyper["niter"]
    hidden = hyper["hidden"] 
    alpha  = hyper["alpha"]
    
    params = fcann2_train(X, 
                          Y_, 
                          param_niter=niter, 
                          param_delta=delta,
                          param_lambda=alpha, 
                          hidden=hidden, 
                          verbose=True, 
                          grad_check=False)
    
    P, Y = fcann2_classify(X, *params)
    bbox=(np.min(X, axis=0), np.max(X, axis=0)) 
    graph_surface(lambda X : fcann2_classify(X, *params)[0][:,1], bbox)
    graph_data(X, Y_, Y, special=[])
    
    C, acc, precisions, recalls  = data.eval_perf_multi(Y,Y_)
    print("----------")
    print(f"acc = {acc}")
    print(f"precission = {precisions}")
    print(f"recallas = {recalls}")
    

if __name__ == "__main__":
    np.random.seed(2)
    X,Y_ = sample_gmm_2d(ncomponents = 4, 
                         nclasses = 2, 
                         nsamples = 30)
    
    params={"alpha" : 0, 
            "delta" : 0.1, 
            "niter" : 10000, 
            "hidden" : 20}
    
    task(X, Y_, params)
# =============================================================================
#     lambdas = np.linspace(0, 1e-1, 100)
#     X,Y_ = sample_gmm_2d(ncomponents = 4, 
#                          nclasses = 2, 
#                          nsamples = 30)
#     for x in lambdas:
#         routine(X,Y_,x)
#         plt.show()
# =============================================================================
    
    
    
    
