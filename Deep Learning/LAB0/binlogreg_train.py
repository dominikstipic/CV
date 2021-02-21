import numpy as np
import data
import matplotlib.pyplot as plt


def gradchecker(grad, fun, w, b):
    approx_grad = []
    EPSILON = 10**-3
    E = np.eye(np.shape(w)[0])
    
    for e in E:
        delta = e*EPSILON
        approx = (fun(w+delta, b) - fun(w-delta, b))/(2*EPSILON)
        approx_grad.append(approx)
    
    approx = (fun(w, b+EPSILON)-fun(w, b-EPSILON))/(2*EPSILON)
    approx_grad.append(approx)
    approx_grad = np.array(approx_grad)
    
    deltas = abs(grad - approx_grad)
    print(deltas)
    
def binlogreg_train(X, Y_, param_niter=100, ni=0.6, alpha=0, gradchecking=False):
    """
      Argumenti
        X: podaci, np.array NxD
        Y_: indeksi razedam, np.array Nx1
        
      Povratne vrijednosti
        w, b: parametri logističke regresije
    """
    N,D = X.shape
    w = np.random.normal(loc=0, scale=1, size=D)
    b = 0
    for i in range(param_niter):
        scores = np.dot(X,w) + b           # N x 1
        probs  = 1/(1+np.exp(-scores))     # N x 1
        # CE
        loss   = -1/N*(sum(Y_*np.log(probs)) + sum((1-Y_)*np.log(1-probs)))
        
        if i % 10 == 0:
            print("iteration {}: loss {}".format(i, loss))
        
        dL_dscores = probs - Y_ 
        grad_w = 1/N * np.dot(dL_dscores, X)
        grad_b = 1/N * sum(dL_dscores)
        if gradchecking:
            fun = lambda w,b : -1/N*(sum(Y_*np.log(1/(1+np.exp(-(np.dot(X,w)+b))))) + sum((1-Y_)*np.log(1-1/(1+np.exp(-(np.dot(X,w)+b))))))
            gradchecker(np.append(grad_w, grad_b), fun, w, b)
            
        w = w*(1-ni*alpha) - ni * grad_w
        b += -ni * grad_b
    
    return w,b
     

def binlogreg_classify(X, w, b):
    """
      Argumenti
        X: podaci, np.array NxD
        w, b: parametri logističke regresije
        
      Povratne vrijednosti
        probs: vjerojatnosti razreda c1
    """
    scores = np.dot(X,w) + b
    probs  = 1/(1+np.exp(-scores))
    return probs


if __name__=="__main__":
    np.random.seed(100)
    
    X,Y_ = data.sample_gauss_2d(2,100)
    w,b  = binlogreg_train(X, Y_, alpha = 0) 
    
    probs = binlogreg_classify(X, w ,b)
    Y     = (probs > 0.5) * 1
    
    accuaracy, precision, recall = data.eval_perf_binary(Y,Y_)
    
    Yr = Y[np.argsort(probs)]
    ap = data.eval_AP(Yr[::-1])
    print("acc={}, precision={}, recall={}, AP={}".format(accuaracy,precision,recall,ap))
    
    # BIN LOG REG
    bbox=(np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(lambda X : binlogreg_classify(X, w, b), bbox, offset=0.5)
    data.graph_data(X, Y_, Y, special=[])
    plt.show()
    
    
    
    


