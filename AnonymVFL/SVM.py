from SharedVariable import SharedVariable
import numpy as np
from common import out_dom

class SVMSS:
    def __init__(self, in_features, lambda_):
        self.w = SharedVariable.from_secret(np.zeros((in_features, 1)))
        self.lambda_ = lambda_

    def forward(self, X : SharedVariable) -> SharedVariable:
        return X @ self.w
    
    def backward(self, X : SharedVariable, y : SharedVariable, y_pred : SharedVariable, lr):
        batch_size = X.shape()[0]
        A_ = np.argwhere(y * y_pred < 1).transpose()[0]
        y = y[A_]
        X = X[A_]
        self.w = (1 - lr * self.lambda_) * self.w + (lr/batch_size) * (X.transpose() @ y)
        pass


from sklearn.metrics import accuracy_score
from tqdm import trange
import matplotlib.pyplot as plt
def train(train_X : SharedVariable, train_y : SharedVariable, test_X, test_y, n_iter = 20000) -> SharedVariable:
    num_samples, num_features = train_X.shape()
    lambda_ = 0.00001
    SS_model = SVMSS(num_features,lambda_)
    batch_size = 16
    accs = []
    max_acc = 0
    for t in trange(1,n_iter+1):

        choice_idx = np.random.choice(num_samples, batch_size, replace=False)
        X = train_X[choice_idx]
        y = train_y[choice_idx]

        y_pred = SS_model.forward(X)

        lr = 1/(lambda_ * t)

        SS_model.backward(X, y, y_pred,lr)


        y_pred = test_X @ SS_model.w.reveal()
        y_pred = np.sign(y_pred)
        
        Accracy = accuracy_score(test_y, y_pred)
        if Accracy > max_acc:
            max_acc = Accracy
        if t % 100 == 0:
            accs.append(Accracy)

    plt.plot(accs,label = "SVM",color="red")
    plt.axhline(max_acc, 0, len(accs), label="Max SVM",color="red")
    plt.legend()
    plt.savefig("SVM.png")
    return SS_model.w

import pandas as pd
from sklearn.model_selection import train_test_split
import os
def SVM_test():
    data = pd.read_csv(os.path.join("Datasets","pcs.csv")).to_numpy()
    train_data, test_data = train_test_split(data)

    train_X = train_data[:,:-1]
    train_X = SharedVariable.from_secret(train_X,out_dom)
    train_y = train_data[:,-1]
    train_y[train_y == 0] = -1
    train_y = SharedVariable.from_secret(train_y.reshape(-1,1),out_dom)

    test_X = test_data[:,:-1]
    test_y = test_data[:,-1]
    test_y[test_y == 0] = -1
    test_y = test_y.reshape(-1,1)
    weight = train(train_X, train_y, test_X,test_y).reveal()

if __name__ == "__main__":
    SVM_test()