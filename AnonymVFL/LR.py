from matplotlib import legend
from SharedVariable import SharedVariable
import numpy as np
from tqdm import trange
from common import out_dom
class LRSS:
    def __init__(self, in_features):
        self.w = SharedVariable(np.zeros((in_features,1)),np.zeros((in_features, 1)))

    @staticmethod
    def activate_fn(x : SharedVariable):
        GT_idx = np.argwhere(x > 1/2)
        LT_idx = np.argwhere(x < -1/2)
        x += 1/2
        for i in GT_idx:
            x[i] = 1
        for i in LT_idx:
            x[i] = 0
        return x

    def forward(self, X : SharedVariable):
        return self.activate_fn(X @ self.w)
    
    def backward(self, X : SharedVariable, y : SharedVariable, y_pred : SharedVariable, lr = 0.1):
        batch_size = X.shape()[0]
        diff = y_pred - y
        delta =  X.transpose() @ diff
        self.w -= (lr/batch_size) * delta
    
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

def train(train_X : SharedVariable, train_y : SharedVariable, test_X, test_y, n_iter = 200) -> SharedVariable:
    num_samples, num_features = train_X.shape()
    # model = LR(num_features, 1)
    SS_model = LRSS(num_features)
    batch_size = 16
    accs = []
    max_acc = 0
    for t in trange(1,n_iter + 1):
        for j in range(0,num_samples,batch_size):
            if j + batch_size > num_samples:
                batch_size = num_samples - j
            X = train_X[j:j+batch_size]
            y = train_y[j:j+batch_size]

            y_pred = SS_model.forward(X)
            SS_model.backward(X, y, y_pred, lr = 0.1 / t)

            y_pred = test_X @ SS_model.w.reveal()
            y_pred = np.clip((y_pred + 1/2),0,1).round()
            Accracy = accuracy_score(test_y, y_pred)
            if Accracy > max_acc:
                max_acc = Accracy
        accs.append(Accracy)
    plt.plot(accs,label = "LR")
    plt.axhline(max_acc, 0, len(accs), label="Max LR")
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("LR.png")
    plt.close()
    return SS_model.w

import pandas as pd
from sklearn.model_selection import train_test_split
import os
def LR_test():
    data = pd.read_csv(os.path.join("Datasets","pcs.csv")).to_numpy()
    train_data, test_data = train_test_split(data)

    train_X = train_data[:,:-1]
    train_X = SharedVariable.from_secret(train_X, out_dom)
    train_y = train_data[:,-1].reshape(-1,1)
    train_y = SharedVariable.from_secret(train_y, out_dom)

    test_X = test_data[:,:-1]
    test_y = test_data[:,-1]
    weight = train(train_X, train_y, test_X,test_y).reveal()

if __name__ == "__main__":
    LR_test()