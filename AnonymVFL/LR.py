import test
from SharedVariable import SharedVariable
import numpy as np
from tqdm import trange, tqdm
from common import out_dom
class LRSS:
    def __init__(self, in_features, out_features = 1, lambda_ = 0):
        self.out_features = out_features
        self.w = SharedVariable(np.zeros((in_features,out_features)),np.zeros((in_features, out_features)))
        self.lambda_ = lambda_

    @staticmethod
    def activate_fn(X : SharedVariable):
        GT_idx = np.argwhere(X > 1/2)
        LT_idx = np.argwhere(X < -1/2)
        X += 1/2
        for i, j in GT_idx:
            X[i, j] = 1
        for i, j in LT_idx:
            X[i, j] = 0
        return X

    def forward(self, X : SharedVariable):
        return self.activate_fn(X @ self.w)
    
    def predict(self, X : np.ndarray):
        y = np.clip((X @ self.w.reveal()) + 1/2, 0, 1).round()
        if self.out_features == 1:
            return y
        else:
            return y.argmax(axis=1).reshape(-1,1)
    
    def backward(self, X : SharedVariable, y : SharedVariable, y_pred : SharedVariable, lr = 0.1):
        batch_size = X.shape()[0]
        diff = y_pred - y
        self.w = (1 - self.lambda_) * self.w - (lr/batch_size) * (X.transpose() @ diff)


import secretflow as sf

sf.init(['company', 'partner', 'coordinator'], address='local')
aby3_config = sf.utils.testing.cluster_def(parties=['company', 'partner', 'coordinator'])
spu = sf.SPU(aby3_config)
company, partner, coordinator = sf.PYU('company'), sf.PYU('partner'), sf.PYU('coordinator')
import jax.numpy as jnp

class SSLR:
    def __init__(self, in_features, out_features = 1, lambda_ = 0):
        self.out_features = out_features
        self.w = spu(jnp.zeros)((in_features, out_features))
        self.lambda_ = lambda_

    def forward(self, X):
        def fw(X, w):
            return jnp.clip(X @ w + 1/2, 0, 1)
        return spu(fw)(X, self.w)

    def predict(self, X):
        y = self.forward(X)
        y = sf.reveal(y)
        if self.out_features == 1:
            return y.round().reshape(-1,1)
        else:
            return y.argmax(axis=1).reshape(-1,1)

    def backward(self, X, y, y_pred, lr = 0.1):
        def grad_desc(lambda_, w, X, y_pred, y):
            batch_size = X.shape[0]
            diff = y_pred - y
            return (1 - lambda_) * w - (lr/batch_size) * (X.transpose() @ diff)
        self.w = spu(grad_desc)(self.lambda_, self.w, X, y_pred, y)


class LR:
    def __init__(self, in_features, out_features = 1, lambda_ = 0, appx_sigmoid = False):
        self.out_features = out_features
        self.w = np.zeros((in_features,out_features))
        self.appx_sigmoid = appx_sigmoid
        self.lambda_ = lambda_

    def activate_fn(self,X : np.ndarray):
        if self.appx_sigmoid:
            X += 1/2
            return np.clip(X,0,1)
        else:
            X = np.clip(X,-500,500)
            if self.out_features ==1:
                return 1/(1 + np.exp(-X))
            return np.exp(X) / np.sum(np.exp(X), axis=1, keepdims=True)

    def forward(self, X : np.ndarray):
        return self.activate_fn(X @ self.w)
    
    def predict(self, X):
        y = self.activate_fn(X @ self.w).round()
        if self.out_features == 1:
            return y
        else:
            return y.argmax(axis=1).reshape(-1,1)

    def backward(self, X : np.ndarray, y : np.ndarray, y_pred : np.ndarray, lr = 0.1):
        batch_size = X.shape[0]
        diff = y_pred - y
        self.w = (1 - self.lambda_) * self.w - (lr/batch_size) * (X.transpose() @ diff)

    def fit(self, X : np.ndarray, y : np.ndarray, n_iter = 100, batch_size = 64):
        num_samples, num_features = X.shape
        for t in range(1,n_iter + 1):
            for j in range(0,num_samples,batch_size):
                batch = min(batch_size,num_samples - j)
                X_batch = X[j:j+batch]
                y_batch = y[j:j+batch]

                y_pred = spu(self.forward)(X_batch)
                spu(self.backward)(X_batch, y_batch, y_pred, lr = 0.1 / t)

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

def train(train_X : SharedVariable, train_y : SharedVariable, test_X, test_y, n_iter = 100, batch_size = 64) -> SharedVariable:
    num_samples, num_features = train_X.shape()
    _, num_cat = train_y.shape()
    
    model = LRSS(num_features, num_cat)
    accs = []
    max_acc = 0
    for t in range(1,n_iter + 1):
        print(f"Epoch {t}")
        for j in trange(0,num_samples,batch_size):
            batch = min(batch_size,num_samples - j)
            X = train_X[j:j+batch]
            y = train_y[j:j+batch]

            y_pred = model.forward(X)
            model.backward(X, y, y_pred, lr = 0.1 / t)

            y_pred = model.predict(test_X)
            Accracy = accuracy_score(test_y, y_pred)
            if Accracy > max_acc:
                max_acc = Accracy
                print(f"Iteration {t}, Batch {j//batch_size + 1}, Accuracy: {Accracy:.4f}")
        accs.append(Accracy)

    plt.plot(accs,label = "LR_SS",color = "blue")
    plt.axhline(max_acc, 0, len(accs), label="Max LR_SS", color = "blue",linestyle = ":")

    train_X = train_X.reveal()
    train_y = train_y.reveal()

    model = LR(num_features, num_cat,appx_sigmoid=True)
    accs = []
    max_acc = 0

    for t in range(1,n_iter + 1):
        print(f"Epoch {t}")
        for j in trange(0,num_samples,batch_size):
            batch = min(batch_size,num_samples - j)
            X = train_X[j:j+batch]
            y = train_y[j:j+batch]

            y_pred = model.forward(X)
            model.backward(X, y, y_pred, lr = 0.1 / t)

            y_pred = model.predict(test_X)
            Accracy = accuracy_score(test_y, y_pred)
            if Accracy > max_acc:
                max_acc = Accracy
        accs.append(Accracy)
    
    plt.plot(accs,label = "LR_without_SS", color = "red",linestyle = "--")
    plt.axhline(max_acc, 0, len(accs), label="Max LR_without_SS", color = "red",linestyle = ":")

    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(max_iter = n_iter,penalty=None)

    if num_cat > 1:
        train_y = train_y.argmax(axis=1)

    model.fit(train_X,train_y.ravel())
    y_pred = model.predict(test_X)
    Accracy = accuracy_score(test_y, y_pred)
    plt.axhline(Accracy, 0, len(accs), label="LR sklearn", color = "green",linestyle = ":")

    plt.xlabel("nIter")
    plt.ylabel("Accuracy")
    plt.legend()


import pandas as pd
from sklearn.model_selection import train_test_split
import os
def load_dataset(dataset : str) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    if dataset == "pima" or dataset == "lbw" or dataset == "pcs" or dataset == "uis":
        data = pd.read_csv(os.path.join("Datasets",f"{dataset}.csv")).to_numpy()
        train_data, test_data = train_test_split(data,shuffle=False)
        train_X = train_data[:,:-1]
        train_y = train_data[:,-1].reshape(-1,1)
        test_X = test_data[:,:-1]
        test_y = test_data[:,-1].reshape(-1,1)

    elif dataset == "gisette" or dataset == "arcene":
        folder = os.path.join("Datasets",dataset)
        train_X = np.loadtxt(os.path.join(folder,f"{dataset}_train.data"))
        train_y = np.loadtxt(os.path.join(folder,f"{dataset}_train.labels"))
        train_y[train_y == -1] = 0
        train_y = train_y.reshape(-1,1)
        test_X = np.loadtxt(os.path.join(folder,f"{dataset}_valid.data"))
        test_y = np.loadtxt(os.path.join(folder,f"{dataset}_valid.labels"))
        test_y[test_y == -1] = 0
        test_y = test_y.reshape(-1,1)

    elif dataset == "mnist":
        from sklearn.datasets import fetch_openml
        mnist = fetch_openml('mnist_784', version=1,as_frame=False)
        X = mnist.data
        y = mnist.target
        train_X, test_X, train_y, test_y = train_test_split(X, y,shuffle=False)
        train_y = train_y.astype(int).reshape(-1,1)
        from sklearn.preprocessing import OneHotEncoder
        train_y = OneHotEncoder().fit_transform(train_y).toarray()
        test_y = test_y.astype(int).reshape(-1,1)
    
    elif dataset == "risk":
        dir_path = os.path.join("Datasets", "data", "data")
        train = pd.read_csv(os.path.join(dir_path, "risk_assessment_all.csv"))
        test = pd.read_csv(os.path.join(dir_path, "risk_assessment_all_test.csv"))
        train_X = train.drop(columns=["id","y"]).to_numpy()
        train_y = train["y"].to_numpy().reshape(-1,1)
        test_X = test.drop(columns=["id","y"]).to_numpy()
        test_y = test["y"].to_numpy().reshape(-1,1)

    elif dataset == "breast":
        dir_path = os.path.join("Datasets", "data", "data")
        guest = pd.read_csv(os.path.join(dir_path, "breast_hetero_guest.csv"))
        host = pd.read_csv(os.path.join(dir_path, "breast_hetero_host.csv"))
        all = pd.concat([host, guest], join = 'inner', axis = 1)
        X = all.drop(columns=["id", "y"]).to_numpy()
        y = all["y"].to_numpy().reshape(-1,1)
        train_X, test_X, train_y, test_y = train_test_split(X, y, shuffle=True)

    return train_X, train_y, test_X, test_y

def LR_test(dataset):
    train_X, train_y, test_X, test_y = load_dataset(dataset)

    train_X = SharedVariable.from_secret(train_X, out_dom)
    train_y = SharedVariable.from_secret(train_y, out_dom)
    train(train_X, train_y, test_X,test_y)
    plt.title(f"LR_{dataset}")
    plt.savefig(f"LR_{dataset}.png")
    plt.close()

def SSLR_test(dataset):
    train_X, train_y, test_X, test_y = load_dataset(dataset)
    test_X = jnp.array(test_X)
    test_X = sf.to(company, test_X).to(spu)
    model = SSLR(train_X.shape[1], train_y.shape[1])
    num_samples = train_X.shape[0]
    num_cat = train_y.shape[1]

    batch_size = 1024
    Xs = []
    ys = []
    for j in trange(0,num_samples,batch_size):
        batch = min(batch_size,num_samples - j)
        X_batch = jnp.array(train_X[j:j+batch])
        X_batch = sf.to(company, X_batch).to(spu)
        y_batch = jnp.array(train_y[j:j+batch])
        y_batch = sf.to(company, y_batch).to(spu)
        Xs.append(X_batch)
        ys.append(y_batch)
    
    n_iter = 20
    accs = []
    max_acc = 0
    for t in range(1,n_iter + 1):
        print(f"Epoch {t}")
        for X,y in tqdm(zip(Xs, ys)):
            y_pred = model.forward(X)
            model.backward(X, y, y_pred, 0.1 / t)

        y_pred = model.predict(test_X)
        Accracy = accuracy_score(test_y, y_pred)
        if Accracy > max_acc:
            max_acc = Accracy
            print(f"Iteration {t}, Accuracy: {Accracy:.4f}")
        accs.append(Accracy)

    plt.plot(accs,label = "SSLR",color = "blue")

    test_X = sf.reveal(test_X)

    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(max_iter = n_iter,penalty=None)

    if num_cat > 1:
        train_y = train_y.argmax(axis=1)

    model.fit(train_X,train_y.ravel())
    y_pred = model.predict(test_X)
    Accracy = accuracy_score(test_y, y_pred)
    plt.axhline(Accracy, 0, len(accs), label="LR sklearn", color = "red",linestyle = "--")

    plt.xlabel("nIter")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title(f"SSLR_{dataset}")
    plt.savefig(f"SSLR_{dataset}.png")
    plt.close()

if __name__ == "__main__":
    # LR_test("mnist")
    # for dataset in ["pima","pcs","uis","gisette","arcene"]:
    #     LR_test(dataset)
    SSLR_test("breast")
    SSLR_test("risk")