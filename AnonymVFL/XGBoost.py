from re import S
import secretflow as sf
import jax.numpy as jnp
from common import sigmoid, SigmoidCrossEntropy, SoftmaxCrossEntropy
from secretflow.device.device.spu import SPUCompilerNumReturnsPolicy
from secretflow.device.device import SPUObject, PYUObject
from secretflow.data import FedNdarray
from secretflow.data.ndarray import load, PartitionWay
from tqdm import tqdm

sf.init(['company', 'partner', 'coordinator'],
        address='local',
        )
aby3_config = sf.utils.testing.cluster_def(parties=['company', 'partner', 'coordinator'])
spu = sf.SPU(aby3_config)
company, partner, coordinator = sf.PYU('company'), sf.PYU('partner'), sf.PYU('coordinator')

class TreeNode:
    def __init__(self, left, right, threshold):
        self.left = left
        self.right = right
        self.threshold = threshold

class Leaf:
    def __init__(self, num):
        self.num = num

class Tree:
    def __init__(self, in_features : int, split_index : int | list[int], out_features : int = 1, lambda_ : float = 1e-5, max_depth : int = 1, div : bool =False):
        self.max_depth : int = max_depth
        self.in_features : int = in_features
        self.out_features : int = out_features
        self.lambda_ : float = lambda_
        self.div : bool = div
        self.split_index = split_index
        self.leaf_weights : list[SPUObject] = []

    def fit(self, X : SPUObject, y : PYUObject, y_pred : PYUObject, buckets : list[list[jnp.ndarray]], SSQuantiles : SPUObject, FedQuantiles : FedNdarray):
        """
        X: Secretly shared input features (in spu)
        y: Plaintext labels at company (in company)
        y_pred: Predicted labels at company (in company)
        buckets: List of buckets for each feature (Public).
        """
        self.SSQuantiles = SSQuantiles
        self.FedQuantiles = FedQuantiles
        loss_fn = SigmoidCrossEntropy if self.out_features == 1 else SoftmaxCrossEntropy
        self.buckets = buckets
        
        def generate_indicator(X : jnp.ndarray) -> jnp.ndarray:
            return jnp.ones((X.shape[0], 1), dtype=int)
        s = spu(generate_indicator)(X)

        def grad_hessian(y : jnp.ndarray, y_pred : jnp.ndarray, loss_fn : SigmoidCrossEntropy | SoftmaxCrossEntropy) -> tuple[jnp.ndarray, jnp.ndarray]:
            g = loss_fn.grad(y, y_pred)
            h = loss_fn.hess(y, y_pred)
            return g, h
        g, h = company(grad_hessian, num_returns=2)(y, y_pred, loss_fn)
        g = g.to(spu)
        h = h.to(spu)
        self.root = self.build_tree(g, h, s, 0)
    def leaf(self, g_sum : SPUObject, h_sum : SPUObject) -> Leaf:
        # if self.div:
        def leaf_weight_div(g_sum : jnp.ndarray, h_sum : jnp.ndarray, lambda_ : float) -> jnp.ndarray:
            return -g_sum / (h_sum + lambda_)
        weight=spu(leaf_weight_div)(g_sum, h_sum, self.lambda_)
        # else:
        #     from jax import random
        #     noise = random.laplace(random.PRNGKey(0)).item()
        #     noisy_hsum = spu(jnp.add)(h_sum, noise)           
        #     lr = 1 / sf.reveal(noisy_hsum)
        #     def leaf_weight_opt(g_sum : jnp.ndarray, h_sum : jnp.ndarray):
        #         n_iter = 10
        #         w = 0
        #         for _ in range(n_iter):
        #             w -= lr * (h_sum * w + g_sum)
        #         return w
        #     weight=spu(leaf_weight_opt)(g_sum, h_sum)
        self.leaf_weights.append(weight)
        return Leaf(len(self.leaf_weights) - 1)
    def build_tree(self, g : SPUObject, h : SPUObject, s : SPUObject, depth : int) -> TreeNode | Leaf:
        def gh_sum(g : jnp.ndarray, h : jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
            """
            Computes the sum of gradients and hessians.
            """
            g_sum = jnp.sum(g)
            h_sum = jnp.sum(h)
            return g_sum, h_sum
        
        g_sum, h_sum = spu(gh_sum,num_returns_policy=SPUCompilerNumReturnsPolicy.FROM_USER, user_specified_num_returns=2)(g, h)

        if depth >= self.max_depth:
            return self.leaf(g_sum, h_sum)
        
        def loss_fraction(g_sum : jnp.ndarray, h_sum : jnp.ndarray, lambda_ : float) -> tuple[jnp.ndarray, jnp.ndarray]:
            loss_n = g_sum * g_sum
            loss_d = h_sum + lambda_
            return loss_n, loss_d

        loss_n, loss_d = spu(loss_fraction, num_returns_policy=SPUCompilerNumReturnsPolicy.FROM_USER, user_specified_num_returns=2)(g_sum, h_sum, self.lambda_)

        G, H = self.aggregate_bucket(g, h)
        
        print("Calculating split info...")
        G_L, G_R, H_L, H_R = [], [], [], []
        
        for j, (G_j, H_j) in enumerate(zip(G, H)):
            print(f"Processing feature {j}/{len(G)}...")
            g_L, h_L = 0, 0
            G_L_j, G_R_j, H_L_j, H_R_j = [], [], [], []
            for g_k, h_k in tqdm(zip(G_j[:-1], H_j[:-1])):
                def gh_LR(g_L : jnp.ndarray, h_L : jnp.ndarray, g_k : jnp.ndarray, h_k : jnp.ndarray, g_sum : jnp.ndarray, h_sum : jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
                    g_L += g_k
                    h_L += h_k
                    g_R = g_sum - g_L
                    h_R = h_sum - h_L
                    return g_L, h_L, g_R, h_R
                g_L, h_L, g_R, h_R = spu(gh_LR, num_returns_policy=SPUCompilerNumReturnsPolicy.FROM_USER, user_specified_num_returns=4)(g_L, h_L, g_k, h_k, g_sum, h_sum)
                
                def GH_LR_j_k(g_L : jnp.ndarray, h_L : jnp.ndarray, g_R : jnp.ndarray, h_R : jnp.ndarray, lambda_ : float) -> tuple[jnp.ndarray, jnp.ndarray]:
                    return g_L * g_L, g_R * g_R, h_L + lambda_, h_R + lambda_
                G_L_j_k, H_L_j_k, G_R_j_k, H_R_j_k = spu(GH_LR_j_k, num_returns_policy=SPUCompilerNumReturnsPolicy.FROM_USER, user_specified_num_returns=4)(g_L, h_L, g_R, h_R, self.lambda_)
                G_L_j.append(G_L_j_k)
                G_R_j.append(G_R_j_k)
                H_L_j.append(H_L_j_k)
                H_R_j.append(H_R_j_k)
            G_L.append(G_L_j)
            G_R.append(G_R_j)
            H_L.append(H_L_j)
            H_R.append(H_R_j)

        j, k, sign = self.split(G_L, G_R, H_L, H_R, loss_n, loss_d)
        if sign:
            threshold = (j, k)
            s_L = spu(jnp.zeros_like)(s)
            s_L = sf.reveal(s_L).astype(bool)
            for bucket in self.buckets[j][:k]:
                s_L |= jnp.zeros_like(s_L).at[bucket].set(True)
            
            s_L = jnp.astype(s_L, jnp.int32)
            s_R = 1 - s_L
            s_L = sf.to(company, s_L).to(spu)
            s_R = sf.to(company, s_R).to(spu)

            def subtree_args(g : jnp.ndarray, h : jnp.ndarray, s : jnp.ndarray, s_L : jnp.ndarray, s_R : jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
                s_L *= s
                s_R *= s                
                g_L = g * s_L
                h_L = h * s_L
                g_R = g * s_R
                h_R = h * s_R
                return g_L, h_L, s_L, g_R, h_R, s_R

            g_L, h_L, s_L, g_R, h_R, s_R = spu(subtree_args, num_returns_policy=SPUCompilerNumReturnsPolicy.FROM_USER, user_specified_num_returns=6)(g, h, s, s_L, s_R)
            branch_L = self.build_tree(g_L, h_L, s_L, depth + 1)
            branch_R = self.build_tree(g_R, h_R, s_R, depth + 1)
            node = TreeNode(left=branch_L, right=branch_R, threshold=threshold)
            return node
        else:
            return self.leaf(g_sum, h_sum)

    def split(self, G_L : list[list[SPUObject]], G_R : list[list[SPUObject]], H_L : list[list[SPUObject]], H_R : list[list[SPUObject]], loss_n : SPUObject, loss_d : SPUObject) -> tuple[int, int, bool]:

        indices = [(j, k) for j in range(len(G_L)) for k in range(len(G_L[j]))]
        def gain(g_L : jnp.ndarray, g_R : jnp.ndarray, h_L : jnp.ndarray, h_R : jnp.ndarray, loss_n : jnp.ndarray, loss_d : jnp.ndarray, lambda_ : float) -> jnp.ndarray:
            return (1/2) * ((g_L / h_L) + (g_R / h_R) - (loss_n / loss_d)) - lambda_
        if self.div:
            # Convert to SPUObject
            G_L_flat = spu(jnp.array)([g for G_j in G_L for g in G_j])
            G_R_flat = spu(jnp.array)([g for G_j in G_R for g in G_j])
            H_L_flat = spu(jnp.array)([h for H_j in H_L for h in H_j])
            H_R_flat = spu(jnp.array)([h for H_j in H_R for h in H_j])
            def argmax_gain(G_L_flat : jnp.ndarray, G_R_flat : jnp.ndarray, H_L_flat : jnp.ndarray, H_R_flat : jnp.ndarray, loss_n : jnp.ndarray, loss_d : jnp.ndarray, lambda_ : float) -> tuple[int, bool]:
                gain_ =  gain(G_L_flat, G_R_flat, H_L_flat, H_R_flat, loss_n, loss_d, lambda_)
                i = jnp.argmax(gain_)
                return i, gain_ > 0
            i, sign = spu(argmax_gain)(G_L_flat, G_R_flat, H_L_flat, H_R_flat, loss_n, loss_d, self.lambda_)
            i = sf.reveal(i)
            sign = sf.reveal(sign)
            j_opt, k_opt = indices[i]
            return j_opt, k_opt, sign
        else:
            def gt(g_L1 : list[float], g_R1 : list[float], h_L1 : float, h_R1 : float, g_L2 : list[float], g_R2 : list[float], h_L2 : list[float], h_R2 : list[float]) -> jnp.ndarray:
                g_L1 = jnp.array(g_L1, dtype=float)
                g_R1 = jnp.array(g_R1, dtype=float)
                g_L2 = jnp.array(g_L2, dtype=float)
                g_R2 = jnp.array(g_R2, dtype=float)
                h_L1 = jnp.array(h_L1, dtype=float)
                h_R1 = jnp.array(h_R1, dtype=float)
                h_L2 = jnp.array(h_L2, dtype=float)
                h_R2 = jnp.array(h_R2, dtype=float)

                h_L12 = h_L1 * h_L2
                h_R12 = h_R1 * h_R2
                nom = h_R12 * (g_L1 * h_L2 - g_L2 * h_L1) + h_L12 * (g_R1 * h_R2 - g_R2 * h_R1)
                denom = h_L12 * h_R12
                return (nom > 0) ^ (denom > 0)

            def argmax(G_L : list[list[SPUObject]], G_R : list[list[SPUObject]], H_L : list[list[SPUObject]], H_R : list[list[SPUObject]]) -> tuple[int, int, float, float, float, float]:
                players = indices.copy()
                # run pairwise elimination until one winner remains
                while len(players) > 1:
                    next_round = []
                    # compare in pairs
                    a = players[0::2]
                    b = players[1::2]
                    if len(players) % 2 == 1:  # if odd number of players, the last one advances automatically
                        next_round.append(a[-1])
                        a.pop()

                    g_La = [G_L[i][j] for i, j in a]
                    g_Ra = [G_R[i][j] for i, j in a]
                    h_La = [H_L[i][j] for i, j in a]
                    h_Ra = [H_R[i][j] for i, j in a]

                    g_Lb = [G_L[i][j] for i, j in b]
                    g_Rb = [G_R[i][j] for i, j in b]
                    h_Lb = [H_L[i][j] for i, j in b]
                    h_Rb = [H_R[i][j] for i, j in b]

                    a_gt_b = spu(gt)(g_La, g_Ra, h_La, h_Ra, g_Lb, g_Rb, h_Lb, h_Rb)
                    a_gt_b = sf.reveal(a_gt_b)
                    next_round.extend(
                        [a_i if a_gt_b[i] else b_i for i, (a_i, b_i) in enumerate(zip(a, b))]
                    )
                    
                    players = next_round

                # the final winner
                j_opt, k_opt = players[0]
                return j_opt, k_opt, G_L[j_opt][k_opt], G_R[j_opt][k_opt], H_L[j_opt][k_opt], H_R[j_opt][k_opt]

            j_opt, k_opt, g_L_opt, g_R_opt, h_L_opt, h_R_opt = argmax(G_L, G_R, H_L, H_R)

            def max_sign(g_L : jnp.ndarray, g_R : jnp.ndarray, h_L : jnp.ndarray, h_R : jnp.ndarray, loss_n : jnp.ndarray, loss_d : jnp.ndarray, lambda_ : float) -> jnp.ndarray:
                # h_LR = h_L * h_R
                # denom = h_LR * loss_n
                # nom = g_L * h_R * loss_d + g_R * h_L * loss_d - loss_n * h_LR - lambda_ * denom
                return gain(g_L, g_R, h_L, h_R, loss_n, loss_d, lambda_) > 0
            sign = spu(max_sign)(g_L_opt, g_R_opt, h_L_opt, h_R_opt, loss_n, loss_d, self.lambda_)
            sign = sf.reveal(sign).item()
            return j_opt, k_opt, sign

    def aggregate_bucket(self, g : SPUObject, h : SPUObject) -> tuple[list[list[SPUObject]], list[list[SPUObject]]]:
        """
        g: Secretly shared gradients (in spu)
        h: Secretly shared hessians (in spu)
        K: Number of buckets to aggregate to (Public).
        """
        
        print("Aggregating buckets...")
        G, H = [], []

        for j, buckets_j in enumerate(self.buckets):
            print(f"Processing feature {j}/{len(self.buckets)}...")
            # Each buckets_j are one-hot encoded, secretly shared vectors for a feature, where each row corresponds to a bucket

            # if len(buckets_j) >= (K+1):
            #     step = len(buckets_j) // (K+1)
            #     temps = []
            #     while len(buckets_j) >= step:
            #         temp = buckets_j[:step]
            #         # flatten temp into a single list of indices
            #         temp = [idx for b in temp for idx in b]
            #         temps.append(sorted(temp))
            #         buckets_j = buckets_j[step:]
            #     if len(buckets_j) > 0:
            #         # If there are remaining buckets, add them as a final bucket
            #         temps.append(sorted([idx for b in buckets_j for idx in b]))
            #     buckets_j = temps
            
            G_j, H_j = [], []
            def bucket_sum(g : jnp.ndarray, h : jnp.ndarray, bucket : jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
                g_sum = jnp.sum(g[bucket])
                h_sum = jnp.sum(h[bucket])
                return g_sum, h_sum
            for bucket in tqdm(buckets_j):
                g_sum, h_sum = spu(bucket_sum, num_returns_policy=SPUCompilerNumReturnsPolicy.FROM_USER, user_specified_num_returns=2)(g, h, bucket)
                G_j.append(g_sum)
                H_j.append(h_sum)
            G.append(G_j)
            H.append(H_j)
        return G, H

    def forward(self, X : FedNdarray | SPUObject) -> PYUObject:
        if isinstance(X, FedNdarray):
            Quantiles = self.FedQuantiles
            mode = 'eval'
        elif isinstance(X, SPUObject):
            Quantiles = self.SSQuantiles
            mode = 'train'
        else:
            raise ValueError("X must be either a FedNdarray or a SPUObject.")
        
        def search_tree(X : FedNdarray | SPUObject, cur : TreeNode | Leaf) -> list[int]:

            def leq(X : jnp.ndarray, j : int, k : int, Quantiles : SPUObject) -> jnp.ndarray:
                return X[:, j] <= Quantiles[j, k]
            
            if mode == 'eval':
                assert isinstance(X, FedNdarray), "X must be a FedNdarray in evaluation mode."
                num_samples = X.shape[0]
                if num_samples == 0:
                    return []
                if isinstance(cur, Leaf):
                    return [cur.num] * num_samples
                elif isinstance(cur, TreeNode):
                    assert isinstance(X, FedNdarray), "X must be a FedNdarray in evaluation mode."
                    j, k = cur.threshold
                    if j < self.split_index:
                        X_c = X.partitions[company]
                        Quantiles_c = Quantiles.partitions[company]
                        X_leq_Qjk = company(leq)(X_c, j, k, Quantiles_c)
                    else:
                        X_p = X.partitions[partner]
                        Quantiles_p = Quantiles.partitions[partner]
                        X_leq_Qjk = partner(leq)(X_p, j - self.split_index, k, Quantiles_p)
                    X_leq_Qjk = sf.reveal(X_leq_Qjk)
                    left_indices = jnp.where(X_leq_Qjk)[0]
                    right_indices = jnp.where(~X_leq_Qjk)[0]
                    X_L = X[left_indices]
                    X_R = X[right_indices]
                    left_results = search_tree(X_L, cur.left)
                    right_results = search_tree(X_R, cur.right)
                    overall_results = [-1] * num_samples
                    for i, idx in enumerate(left_indices):
                        overall_results[idx] = left_results[i]
                    for i, idx in enumerate(right_indices):
                        overall_results[idx] = right_results[i]
                    return overall_results

            elif mode == 'train':
                assert isinstance(X, SPUObject), "X must be a SPUObject in training mode." 
                num_samples = sf.reveal(spu(jnp.shape)(X))[0].item()
                if num_samples == 0:
                    return []
                if isinstance(cur, Leaf):
                    return [cur.num] * num_samples
                elif isinstance(cur, TreeNode):
                    j, k = cur.threshold

                    X_leq_Qjk = spu(leq)(X, j, k, Quantiles)
                    X_leq_Qjk = sf.reveal(X_leq_Qjk)
                    left_indices = jnp.where(X_leq_Qjk)[0]
                    right_indices = jnp.where(~X_leq_Qjk)[0]
                    def X_LR(X : jnp.ndarray, left_indices : jnp.ndarray, right_indices : jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
                        return X[left_indices], X[right_indices]
                    X_L, X_R = spu(X_LR, num_returns_policy=SPUCompilerNumReturnsPolicy.FROM_USER, user_specified_num_returns=2)(X, left_indices, right_indices)
                    left_results = search_tree(X_L, cur.left)
                    right_results = search_tree(X_R, cur.right)
                    overall_results = [-1] * num_samples
                    for i, idx in enumerate(left_indices):
                        overall_results[idx] = left_results[i]
                    for i, idx in enumerate(right_indices):
                        overall_results[idx] = right_results[i]
                    return overall_results
            else:
                raise ValueError("Mode must be 'train' or 'eval'.")
            
        leaves_ids = search_tree(X, self.root)
        w = [self.leaf_weights[leaf_id] for leaf_id in leaves_ids]
        def ret(w : list[jnp.ndarray]):
            return jnp.array(w, dtype=jnp.float32).reshape(-1,1)
        return spu(ret)(w).to(company)

class SSXGBoost:
    def __init__(self,in_features, split_index : int | list[int], out_features = 1, n_estimators = 3, lambda_ = 1e-5, max_depth = 3, div = False):
        self.trees : list[Tree] = []
        self.in_features = in_features
        self.out_features = out_features
        self.n_estimators = n_estimators
        self.lambda_ = lambda_
        self.max_depth = max_depth
        self.div = div
        self.split_index = split_index 

    def forward(self, X : FedNdarray | SPUObject):
        preds = 0
        for m in self.trees:
            preds = company(jnp.add)(preds, m.forward(X))
        return preds

    def predict(self, X : FedNdarray):
        y = self.forward(X)
        y = sf.reveal(y)
        y = sigmoid(y)
        if self.out_features == 1:
            return y.round()
        else:
            return y.argmax(axis=1).reshape(-1,1)

    def fit(self, X : SPUObject, y : PYUObject, buckets : list[list[jnp.ndarray]], SSQuantiles : SPUObject, FedQuantiles : FedNdarray):
        """
        X: Secretly shared input features (in spu)
        y: Plaintext labels at company (in company)
        buckets: List of buckets for each feature (public).
        Quantiles: List of quantiles for each feature (horizontally partitioned).
        """
        self.SSQuantiles = SSQuantiles
        self.FedQuantiles = FedQuantiles

        y_pred = company(jnp.zeros_like)(y)
        for _ in range(self.n_estimators):
            tree = Tree(self.in_features, self.split_index, self.out_features, self.lambda_, self.max_depth, self.div)
            tree.fit(X, y, y_pred, buckets, self.SSQuantiles, self.FedQuantiles)

            y_t = tree.forward(X)
            y_pred = company(jnp.add)(y_pred, y_t)
            self.trees.append(tree)

import pandas as pd
from sklearn.model_selection import train_test_split
import os
import numpy as np
from sklearn.metrics import accuracy_score
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

def SSXGBoost_test(dataset):
    train_X, train_y, test_X, test_y = load_dataset(dataset)


    # 将 train_X 每列等频分桶为 k+1 份，并计算 k 个分位点
    split_index = train_X.shape[1] // 2
    k = 63   # 把每列分成 k+1 份，所以要算 k 个分位点
    buckets, Quantiles = [], []

    for j in range(train_X.shape[1]):
        col = train_X[:, j]
        # 1) 计算分位点
        qs = np.quantile(col, [(i + 1) / (k + 1) for i in range(k)]).round(3)
        Quantiles_j = []
        # 2) 排序后等分索引
        buckets_j = []
        left = float('-inf')
        right = qs[0]
        for i in range(len(qs)):
            # 计算每个分位点对应的索引范围
            indices = jnp.where((col > left) & (col <= right))[0]
            if len(indices) > 0:
                Quantiles_j.append(right)
                buckets_j.append(indices.sort())
            left = right
            right = float('inf') if i == len(qs) - 1 else qs[i + 1]
        # 3) 最后一个分位点对应的索引范围
        indices = jnp.where(col > left)[0]
        if len(indices) > 0:
            buckets_j.append(indices.sort())

        Quantiles.append(Quantiles_j)
        buckets.append([g for g in buckets_j])

    Quantiles = np.array(Quantiles)

    
    Quantiles1, Quantiles2 = Quantiles[:split_index], Quantiles[split_index:]
    np.save("Quantiles1.npy", Quantiles1)
    np.save("Quantiles2.npy", Quantiles2)
    SSQuantiles = sf.to(company, jnp.array(Quantiles)).to(spu)
    FedQuantiles = load({company: "Quantiles1.npy", partner: "Quantiles2.npy"}, partition_way=PartitionWay.HORIZONTAL)

    model = SSXGBoost(train_X.shape[1], split_index, out_features= train_y.shape[1])
    num_samples, num_cat = train_X.shape[0], train_y.shape[1]

    # 然后把训练集 secret‐share 到 SPU
    train_X = sf.to(company, jnp.array(train_X)).to(spu)
    train_y = sf.to(company, jnp.array(train_y,dtype=jnp.float32))

    test_X1, test_X2 = test_X[:, split_index:], test_X[:, :split_index]
    np.save("test_X1.npy", test_X1)
    np.save("test_X2.npy", test_X2)
    test_X = load({company: "test_X1.npy", partner: "test_X2.npy"})

    from os import remove
    remove("test_X1.npy")
    remove("test_X2.npy")
    remove("Quantiles1.npy")
    remove("Quantiles2.npy")

    model.fit(train_X, train_y, buckets, SSQuantiles, FedQuantiles)

    y_pred = model.predict(test_X)
    Accuracy = accuracy_score(test_y, y_pred)
    print(f"Accuracy of SSXGBoost on {dataset} dataset: {Accuracy:.4f}")

    test_X = sf.reveal(test_X)

    # import xgboost as xgb
    # model = xgb.XGBClassifier()

    # if num_cat > 1:
    #     train_y = train_y.argmax(axis=1)

    # model.fit(train_X,train_y.ravel())
    # y_pred = model.predict(test_X)
    # Accuracy = accuracy_score(test_y, y_pred)
    # print(f"Accuracy of XGBoost on {dataset} dataset: {Accuracy:.4f}")

if __name__ == "__main__":
    SSXGBoost_test("breast")