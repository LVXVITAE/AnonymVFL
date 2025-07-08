from SharedVariable import SharedVariable
import numpy as np
from common import out_dom

class SVMSS:
    def __init__(self, in_features, out_features =1, lambda_ = 0.00001):
        self.out_features = out_features
        self.w = SharedVariable.from_secret(np.zeros((in_features, out_features)))
        self.lambda_ = lambda_

    def forward(self, X : SharedVariable) -> SharedVariable:
        return X @ self.w
    
    def predict(self, X : np.ndarray):
        y = X @ self.w.reveal()
        if self.out_features == 1:
            return np.sign(y)
        else:
            return y.argmax(axis=1).reshape(-1,1)
    
    def backward(self, X : SharedVariable, y : SharedVariable, y_pred : SharedVariable, lr):
        batch_size = X.shape()[0]
        LT1 = (y * y_pred < 1).T
        A_ = [np.argwhere(A_i).ravel() for A_i in LT1]

        for j in range(self.out_features):
            Xj = X[A_[j]]
            yj = y[A_[j],j]
            self.w[:,j] = (1 - lr * self.lambda_) * self.w[:,j] + (lr/batch_size) * (Xj.transpose() @ yj)
        

class SVM:
    def __init__(self, in_features, out_features = 1, lambda_ = 0.00001):
        self.out_features = out_features
        self.w = np.zeros((in_features, out_features))
        self.lambda_ = lambda_

    def forward(self, X : np.ndarray) -> np.ndarray:
        return X @ self.w
    
    def predict(self, X : np.ndarray):
        y = X @ self.w
        if self.out_features == 1:
            return np.sign(y)
        else:
            return y.argmax(axis=1).reshape(-1,1)
    
    def backward(self, X : np.ndarray, y : np.ndarray, y_pred : np.ndarray, lr):
        batch_size = X.shape[0]
        LT1 = (y * y_pred < 1).T
        A_ = [np.argwhere(A_i).ravel() for A_i in LT1]

        for j in range(self.out_features):
            Xj = X[A_[j]]
            yj = y[A_[j],j]
            self.w[:,j] = (1 - lr * self.lambda_) * self.w[:,j] + (lr/batch_size) * (Xj.transpose() @ yj)

from sklearn.metrics import accuracy_score
from tqdm import trange
import matplotlib.pyplot as plt
def train(train_X : SharedVariable, train_y : SharedVariable, test_X, test_y, n_iter = 5000, batch_size = 64) -> SharedVariable:
    num_samples, num_features = train_X.shape()
    _, num_cat = train_y.shape()
    model = SVMSS(num_features,num_cat)
    iters = []
    accs = []
    max_acc = 0
    for t in trange(1,n_iter+1):

        choice_idx = np.random.choice(num_samples, batch_size, replace=False)
        X = train_X[choice_idx]
        y = train_y[choice_idx]

        y_pred = model.forward(X)

        lr = 1/(model.lambda_ * t)

        model.backward(X, y, y_pred,lr)


        y_pred = model.predict(test_X)
        
        Accracy = accuracy_score(test_y, y_pred)
        if Accracy > max_acc:
            max_acc = Accracy
        if t % 50 == 0:
            iters.append(t)
            accs.append(Accracy)

    plt.plot(iters,accs,label = "SVM_SS",color = "blue")
    plt.axhline(max_acc, 0, len(accs), label="Max SVM_SS", color = "blue",linestyle = ":")

    train_X = train_X.reveal()
    train_y = train_y.reveal()

    model = SVM(num_features,num_cat)
    iters = []
    accs = []
    max_acc = 0
    for t in trange(1,n_iter+1):

        choice_idx = np.random.choice(num_samples, batch_size, replace=False)
        X = train_X[choice_idx]
        y = train_y[choice_idx]

        y_pred = model.forward(X)

        lr = 1/(model.lambda_ * t)

        model.backward(X, y, y_pred,lr)


        y_pred = model.predict(test_X)
        
        Accracy = accuracy_score(test_y, y_pred)
        if Accracy > max_acc:
            max_acc = Accracy
        if t % 50 == 0:
            iters.append(t)
            accs.append(Accracy)

    plt.plot(iters,accs,label = "SVM_without_SS",color = "red",linestyle = "--")
    plt.axhline(max_acc, 0, len(accs), label="Max SVM_without_SS", color = "red",linestyle = ":")

    from sklearn.svm import LinearSVC

    model = LinearSVC(max_iter = n_iter // batch_size)

    if num_cat > 1:
        train_y = train_y.argmax(axis=1)
    model.fit(train_X,train_y.ravel())
    y_pred = model.predict(test_X)
    Accracy = accuracy_score(test_y, y_pred)
    plt.axhline(Accracy, 0, len(accs), label="SVM sklearn", color = "green",linestyle = ":")

    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.legend()

import pandas as pd
from sklearn.model_selection import train_test_split
import os

def load_dataset(dataset : str) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    if dataset == "pima" or dataset == "lbw" or dataset == "pcs" or dataset == "uis":
        data = pd.read_csv(os.path.join("Datasets",f"{dataset}.csv")).to_numpy()
        data = pd.read_csv(os.path.join("Datasets",f"{dataset}.csv")).to_numpy()
        train_data, test_data = train_test_split(data,shuffle=False)
        train_X = train_data[:,:-1]
        train_y = train_data[:,-1]
        train_y[train_y == 0] = -1
        train_y = train_y.reshape(-1,1)
        test_X = test_data[:,:-1]
        test_y = test_data[:,-1]
        test_y[test_y == 0] = -1
        test_y = test_y.reshape(-1,1)

    elif dataset == "gisette" or dataset == "arcene":
        folder = os.path.join("Datasets",dataset)
        train_X = np.loadtxt(os.path.join(folder,f"{dataset}_train.data"))
        train_y = np.loadtxt(os.path.join(folder,f"{dataset}_train.labels"))
        train_y = train_y.reshape(-1,1)
        test_X = np.loadtxt(os.path.join(folder,f"{dataset}_valid.data"))
        test_y = np.loadtxt(os.path.join(folder,f"{dataset}_valid.labels"))

    elif dataset == "mnist":
        from sklearn.datasets import fetch_openml
        mnist = fetch_openml('mnist_784', version=1,as_frame=False)
        X = mnist.data
        y = mnist.target
        train_X, test_X, train_y, test_y = train_test_split(X, y,shuffle=False)
        train_y = train_y.astype(int).reshape(-1,1)
        from sklearn.preprocessing import OneHotEncoder
        train_y = OneHotEncoder().fit_transform(train_y).toarray()
        train_y[train_y == 0] = -1
        test_y = test_y.astype(int).reshape(-1,1)
        test_y = test_y.reshape(-1,1)

    return train_X, train_y, test_X, test_y

def SVM_test(dataset):
    train_X, train_y, test_X, test_y = load_dataset(dataset)   
    train_X = SharedVariable.from_secret(train_X,out_dom)
    train_y = SharedVariable.from_secret(train_y,out_dom)
    train(train_X, train_y, test_X,test_y)
    plt.title(f"SVM_{dataset}")
    plt.savefig(f"SVM_{dataset}.png")
    plt.close()

if __name__ == "__main__":
    SVM_test("mnist")
    for dataset in ["pima","pcs","uis","gisette","arcene"]:
        SVM_test(dataset)