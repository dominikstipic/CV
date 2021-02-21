import numpy as np
import data

def split_test_valid(x_train, y_train):
    N_valid = int(1/5 * len(x_train))
    valid_mask = np.random.choice(np.arange(len(x_train)), N_valid, replace=False)
    train_mask = list(set(np.arange(len(x_train))).difference(set(valid_mask)))
    x_valid, y_valid = x_train[valid_mask], y_train[valid_mask]
    x_train, y_train = x_train[train_mask], y_train[train_mask]
    return x_train, y_train, x_valid, y_valid

def get_batches(x_train, y_train, batch_size):
    x_dim = x_train.shape[1]
    y_dim = y_train.shape[1]
    x_batches = x_train.view(-1, batch_size, x_dim)
    y_batches = y_train.view(-1, batch_size, y_dim)
    return x_batches, y_batches 

def evaluate(model, X, Y_, P):
    Y = np.argmax(P,axis=1)
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