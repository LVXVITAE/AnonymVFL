import numpy as np
import secretflow as sf
import jax.numpy as jnp
from common import SS_share, sigmoid, approx_sigmoid, SigmoidCrossEntropy, ApproxSigmoidCrossEntropy, SoftmaxCrossEntropy, MeanSquare, softmax, to_int_labels, cross_entropy, mean_square_error, compute_accuracy
from secretflow.device.device.spu import SPUCompilerNumReturnsPolicy
from secretflow.device import SPUObject, PYUObject
from secretflow import SPU, PYU
from secretflow.data import FedNdarray
from secretflow.data.ndarray import load, PartitionWay
from tqdm.contrib import tzip
from tqdm import tqdm
import dill 
import json

class TreeNode:
    """
    树的节点，包括两个左右子树（类型为TreeNode或Leaf）和节点分裂的阈值
    """
    def __init__(self, left, right, threshold : tuple[int, int]):
        """
        ## Args:
        - left: 左子树（类型为TreeNode或Leaf）
        - right: 右子树（类型为TreeNode或Leaf）
        - threshold: 分裂的阈值，类型为tuple[int, int]，对应分位点的索引
        """
        self.left = left
        self.right = right
        self.threshold = threshold
        self.type = 'node'

class Leaf:
    """叶子节点，包含叶子权重的索引，叶子权重以数组的形式保存在Tree类中"""
    def __init__(self, num):
        """
        ## Args:
        - num: 叶子权重的索引，类型为int
        """
        self.num = num
        self.type = 'leaf'

class Tree:
    def __init__(self, devices : dict, lambda_ : float = 1e-5, max_depth : int = 3, div : bool =False, mission = 'Classification'):
        """
        初始化树
        ## Args:
         - devices : 应包含四个字段，每个字段的值应为SPU或PYU。例如：

           devices = {
            'spu': spu,
            'company': company,
            'partner': partner,
           }
        - lambda_: l2正则化参数，默认为1e-5
        - max_depth: 树的最大深度，默认为3
        - div: 是否使用除法。如果为True，则使用除法计算叶子权重和信息增益；如果为False，则使用优化算法计算叶子权重和增益最大分裂点。
        """
        self.max_depth : int = max_depth
        self.lambda_ : float = lambda_
        self.div : bool = div
        # 叶子权重列表，存储每个叶子节点的权重
        self.leaf_weights = []

        self.company : PYU = devices['company']
        self.partner : PYU = devices['partner']
        self.spu : SPU = devices['spu']
        self.mission = mission

    def fit(self, X : SPUObject, y : PYUObject | SPUObject, y_pred : PYUObject | SPUObject, buckets : np.ndarray, FedQuantiles : FedNdarray):
        """
        ## Args:
         - X: 秘密共享的输入特征
         - y: 明文标签， 由label_holder持有
         - y_pred: 明文预测标签， 由label_holder持有
         - buckets: 桶列表（公开）。每个元素bucket_j是特征j的桶列表。bucket_j中的每个桶是一个一维数组，表示桶内元素在X中的索引。
        """
        self.train_label_keeper : PYU|SPU = y.device
        assert X.device == self.spu, "X must be on SPU of the model."
        self.FedQuantiles : FedNdarray = FedQuantiles
        
        self.buckets = buckets

        self.split_index = sf.reveal(self.FedQuantiles.partition_shape()[self.company])[0]

        self.num_train_samples, self.in_features = sf.reveal(self.spu(jnp.shape)(X))
        _, self.out_features = sf.reveal(self.train_label_keeper(jnp.shape)(y))
        self.in_features = int(self.in_features)
        self.out_features = int(self.out_features)
        # 生成指示向量。指示向量是一个01向量，1表示该数据属于本树节点
        def generate_indicator(X : jnp.ndarray) -> jnp.ndarray:
            return jnp.ones((X.shape[0], 1), dtype=int)
        s = self.spu(generate_indicator)(X)
        #计算一阶二阶梯度
        if self.mission == 'Classification':
            if self.out_features == 1:
                if self.train_label_keeper == self.spu:
                    loss_fn = ApproxSigmoidCrossEntropy()
                else:
                    loss_fn = SigmoidCrossEntropy()  
            else: 
                loss_fn = SoftmaxCrossEntropy()
        elif self.mission == 'Regression':
            loss_fn = MeanSquare()

        g = self.train_label_keeper(loss_fn.grad)(y, y_pred).to(self.spu)
        h = self.train_label_keeper(loss_fn.hess)(y, y_pred).to(self.spu)
        self.train_pred = 0.0
        self.root = self._build_tree(g, h, s, 0)
        self.leaf_weights = self.spu(lambda x: jnp.array(x))(self.leaf_weights)
        return
    
    def __reveal_list(self, arr : list):
        '''DEBUG ONLY'''
        def to_jnp(arr : list):
            arr = jnp.array(arr)
            return arr
        arr = self.spu(to_jnp)(arr)
        return sf.reveal(arr)

    def _leaf(self, g_sum : SPUObject, h_sum : SPUObject, s : SPUObject) -> Leaf:
        """
        计算叶子节点的权重。目前只实现了除法版本，尚未实现优化算法版本。
        ## Args:
        - g_sum: 属于本叶子节点的一阶梯度的总和，类型为SPUObject
        - h_sum: 属于本叶子节点的二阶梯度的总和，类型为SPUObject
        ## Returns:
        - Leaf: 叶子节点，包含叶子权重的索引
        """
        # if self.div:
        def leaf_weight_div(g_sum : jnp.ndarray, h_sum : jnp.ndarray, lambda_ : float) -> jnp.ndarray:
            return -g_sum / (h_sum + lambda_)
        weight=self.spu(leaf_weight_div)(g_sum, h_sum, self.lambda_)
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

        self.leaf_weights.append(weight)    # 在叶子权重列表中添加新叶子权重
        def update_pred(pred : jnp.ndarray, weight : jnp.ndarray, s : jnp.ndarray):
            return pred + weight * s
        self.train_pred = self.spu(update_pred)(self.train_pred, weight, s)
        return Leaf(len(self.leaf_weights) - 1)
    def _build_tree(self, g : SPUObject, h : SPUObject, s : SPUObject, depth : int) -> TreeNode | Leaf:
        """
        构建子树
        ## Args:
        - g: 本子树节点的一阶梯度，类型为SPUObject。非本节点的数据其对应一阶梯度为0。
        - h: 本子树节点的二阶梯度，类型为SPUObject。非本节点的数据其对应二阶梯度为0。
        - s: 本子树节点的指示向量，类型为SPUObject。本节点数据在指示向量中用1表示，非本节点的数据用0表示。
        - depth: 当前树的深度，类型为int
        """
        def gh_sum(g : jnp.ndarray, h : jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
            """
            本节点一阶二阶梯度之和
            """
            g_sum = jnp.sum(g)
            h_sum = jnp.sum(h)
            return g_sum, h_sum
        
        g_sum, h_sum = self.spu(gh_sum,num_returns_policy=SPUCompilerNumReturnsPolicy.FROM_USER, user_specified_num_returns=2)(g, h)

        if depth >= self.max_depth:
            return self._leaf(g_sum, h_sum, s)
        
        def loss_fraction(g_sum : jnp.ndarray, h_sum : jnp.ndarray, lambda_ : float) -> tuple[jnp.ndarray, jnp.ndarray]:
            """ 计算当前节点的目标损失的分子和分母"""
            loss_n = g_sum * g_sum
            loss_d = h_sum + lambda_
            return loss_n, loss_d

        loss_n, loss_d = self.spu(loss_fraction, num_returns_policy=SPUCompilerNumReturnsPolicy.FROM_USER, user_specified_num_returns=2)(g_sum, h_sum, self.lambda_)

        G, H = self._aggregate_bucket(g, h)

        # 计算每个分裂点左右的一阶二阶梯度和
        print("Calculating gradient split info...")
        G_L, G_R, H_L, H_R = [], [], [], []

        def split_info(g_L : jnp.ndarray, h_L : jnp.ndarray, g_k : jnp.ndarray, h_k : jnp.ndarray, g_sum : jnp.ndarray, h_sum : jnp.ndarray, lambda_ : float):
            g_L += g_k
            h_L += h_k
            g_R = g_sum - g_L
            h_R = h_sum - h_L
            return g_L, h_L, g_R, h_R, g_L * g_L, h_L + lambda_, g_R * g_R, h_R + lambda_

        for G_j, H_j in tzip(G, H): 
            g_L, h_L = 0.0, 0.0
            G_L_j, G_R_j, H_L_j, H_R_j = [], [], [], []
            for g_k, h_k in zip(G_j[:-1], H_j[:-1]):
                g_L, h_L, g_R, h_R, G_L_j_k, H_L_j_k, G_R_j_k, H_R_j_k = self.spu(
                    split_info, 
                    num_returns_policy=SPUCompilerNumReturnsPolicy.FROM_USER, 
                    user_specified_num_returns=8
                )(g_L, h_L, g_k, h_k, g_sum, h_sum, self.lambda_ )
                
                G_L_j.append(G_L_j_k)
                G_R_j.append(G_R_j_k)
                H_L_j.append(H_L_j_k)
                H_R_j.append(H_R_j_k)
            G_L.append(G_L_j)
            G_R.append(G_R_j)
            H_L.append(H_L_j)
            H_R.append(H_R_j)

        # 计算最优分裂点索引，以及最优分裂点增益的正负
        j, k, sign = self._split(G_L, G_R, H_L, H_R, loss_n, loss_d)
        if sign:
            # 阈值是某个分位点。为了方便存储，这里保存分位点的索引
            threshold = (j, k)
            # 小于阈值的节点指示向量
            s_L = np.zeros((self.num_train_samples, 1))
            for bucket in self.buckets[j][:k+1]:
                s_L[bucket] = 1.0

            # 大于等于阈值的节点指示向量
            s_R = 1.0 - s_L

            if isinstance(self.train_label_keeper, PYU):
                s_L = sf.to(self.train_label_keeper, s_L).to(self.spu)
                s_R = sf.to(self.train_label_keeper, s_R).to(self.spu)
            elif self.train_label_keeper == self.spu:
                s_L = sf.to(self.company, s_L).to(self.spu)
                s_R = sf.to(self.company, s_R).to(self.spu)

            def subtree_args(g : jnp.ndarray, h : jnp.ndarray, s : jnp.ndarray, s_L : jnp.ndarray, s_R : jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
                """
                计算左子树和右子树的一阶二阶梯度以及指示向量
                """
                s_L *= s
                s_R *= s                
                g_L = g * s_L
                h_L = h * s_L
                g_R = g * s_R
                h_R = h * s_R
                return g_L, h_L, s_L, g_R, h_R, s_R

            g_L, h_L, s_L, g_R, h_R, s_R = self.spu(subtree_args, num_returns_policy=SPUCompilerNumReturnsPolicy.FROM_USER, user_specified_num_returns=6)(g, h, s, s_L, s_R)
            branch_L = self._build_tree(g_L, h_L, s_L, depth + 1)
            branch_R = self._build_tree(g_R, h_R, s_R, depth + 1)
            node = TreeNode(left=branch_L, right=branch_R, threshold=threshold)
            return node
        else:
            return self._leaf(g_sum, h_sum, s)

    def _split(self, G_L : list[list[SPUObject]], G_R : list[list[SPUObject]], H_L : list[list[SPUObject]], H_R : list[list[SPUObject]], loss_n : SPUObject, loss_d : SPUObject) -> tuple[int, int, bool]:
        """
        计算最优分裂点索引，以及最优分裂点增益的正负
        ## Args:
        - G_L: 小于某分裂点的数据对应的一阶梯度之和列表
        - G_R: 大于等于某分裂点的数据对应的一阶梯度之和列表
        - H_L: 小于某分裂点的数据对应的二阶梯度之和列表
        - H_R: 大于等于某分裂点的数据对应的二阶梯度之和列表
        - loss_n: 当前节点的目标损失的分子
        - loss_d: 当前节点的目标损失的分母
        """
        indices = [(j, k) for j in range(len(G_L)) for k in range(len(G_L[j]))]
        print("Selecting best split ...")
        def gain(g_L : jnp.ndarray, g_R : jnp.ndarray, h_L : jnp.ndarray, h_R : jnp.ndarray, loss_n : jnp.ndarray, loss_d : jnp.ndarray, lambda_ : float) -> jnp.ndarray:
            """
            计算增益
            ## Args:
            - g_L: 小于本分裂点的数据的对应一阶梯度之和
            - g_R: 大于等于本分裂点的数据的对应一阶梯度之和
            - h_L: 小于本分裂点的数据的对应二阶梯度之和
            - h_R: 大于等于本分裂点的数据的对应二阶梯度之和
            - loss_n: 当前节点的目标损失的分子
            - loss_d: 当前节点的目标损失的分母
            """
            return (1/2) * ((g_L / h_L) + (g_R / h_R) - (loss_n / loss_d)) - lambda_
        if self.div: # 使用除法直接计算所有节点的增益（尚未测试）
            def argmax_gain(G_L : jnp.ndarray, G_R : jnp.ndarray, H_L : jnp.ndarray, H_R : jnp.ndarray, loss_n : jnp.ndarray, loss_d : jnp.ndarray, lambda_ : float) -> tuple[int, bool]:
                """
                计算增益最大值
                """
                G_L = jnp.array(G_L)
                G_R = jnp.array(G_R)
                H_L = jnp.array(H_L)
                H_R = jnp.array(H_R)
                gain_ =  gain(G_L, G_R, H_L, H_R, loss_n, loss_d, lambda_)
                i = jnp.argmax(gain_)
                return i, gain_.flatten()[i] > 0
            i, sign = self.spu(argmax_gain,num_returns_policy=SPUCompilerNumReturnsPolicy.FROM_USER, user_specified_num_returns=2)(G_L, G_R, H_L, H_R, loss_n, loss_d, self.lambda_)
            i = sf.reveal(i)
            sign = sf.reveal(sign).item()
            j_opt, k_opt = indices[i]
            return j_opt, k_opt, sign
        else:
            def leq(g_L1 : list[float], g_R1 : list[float], h_L1 : list[float], h_R1 : list[float], g_L2 : list[float], g_R2 : list[float], h_L2 : list[float], h_R2 : list[float]) -> jnp.ndarray:
                """
                比较两个分裂点的增益。本函数可以拓展为向量以并行地比较多对分裂点的增益
                ## Args:
                - g_L1: 小于第一个分裂点的数据对应一阶梯度之和
                - g_R1: 大于等于第一个分裂点的数据对应一阶梯度之和
                - h_L1: 小于第一个分裂点的数据对应二阶梯度之和
                - h_R1: 大于等于第一个分裂点的数据对应二阶梯度之和
                - g_L2: 小于第二个分裂点的数据对应一阶梯度之和
                - g_R2: 大于等于第二个分裂点的数据对应一阶梯度之和
                - h_L2: 小于第二个分裂点的数据对应二阶梯度之和
                - h_R2: 大于等于第二个分裂点的数据对应二阶梯度之和
                ## Returns:
                - 第一个分裂点的增益是否大于第二个分裂点的增益
                """
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
                """
                使用分组两两比较的方法求解增益最大分裂点
                ## Returns:
                - j_opt: 最优分裂点的特征索引
                - k_opt: 最优分裂点的分位点索引
                - g_L_opt: 最优分裂点左侧的一阶梯度之和
                - g_R_opt: 最优分裂点右侧的一阶梯度之和
                - h_L_opt: 最优分裂点左侧的二阶梯度之和
                - h_R_opt: 最优分裂点右侧的二阶梯度之和
                """
                players = indices.copy()

                while len(players) > 1:
                    next_round = []
                    # 分两组
                    a = players[0::2]
                    b = players[1::2]
                    if len(players) % 2 == 1:  # 如果有奇数个候选，最后一个放到下一轮比较
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

                    a_lt_b = self.spu(leq)(g_La, g_Ra, h_La, h_Ra, g_Lb, g_Rb, h_Lb, h_Rb)
                    a_lt_b = sf.reveal(a_lt_b)

                    next_round.extend(
                        [b_i if a_lt_b[i] else a_i for i, (a_i, b_i) in enumerate(zip(a, b))]
                    )
                    
                    players = next_round

                # the final winner
                j_opt, k_opt = players[0]
                return j_opt, k_opt

            j_opt, k_opt = argmax(G_L, G_R, H_L, H_R)
            g_L_opt = G_L[j_opt][k_opt]
            g_R_opt = G_R[j_opt][k_opt]
            h_L_opt = H_L[j_opt][k_opt]
            h_R_opt = H_R[j_opt][k_opt]

            def max_gain_sign(g_L : jnp.ndarray, g_R : jnp.ndarray, h_L : jnp.ndarray, h_R : jnp.ndarray, loss_n : jnp.ndarray, loss_d : jnp.ndarray, lambda_ : float) -> jnp.ndarray:
                """ 计算增益最大分裂点的正负"""
                h_LR = h_L * h_R
                denom = 2 * h_LR * loss_d
                nom = (g_L * h_R + h_L * g_R - 2 * lambda_ * h_LR) * loss_d - h_LR * loss_n
                return ~ (nom > 0) ^ (denom > 0)
            # max_gain = self.spu(gain)(g_L_opt, g_R_opt, h_L_opt, h_R_opt, loss_n, loss_d, self.lambda_)
            # max_gain = sf.reveal(max_gain)
            sign = self.spu(max_gain_sign)(g_L_opt, g_R_opt, h_L_opt, h_R_opt, loss_n, loss_d, self.lambda_)
            sign = sf.reveal(sign).item()
            return j_opt, k_opt, sign

    def _aggregate_bucket(self, g : SPUObject, h : SPUObject) -> tuple[list[list[SPUObject]], list[list[SPUObject]]]:
        """
        求每个桶内的一阶梯度和二阶梯度之和
        ## Args:
        - g: 本节点一阶梯度
        - h: 本节点二阶梯度
        ## Returns:
        - G: 每个桶内的一阶梯度之和列表，和桶列表的形状相同
        - H: 每个桶内的二阶梯度之和列表，和桶列表的形状相同
        """
        
        print("Aggregating buckets for each feature...")

        def bucket_sum(g : jnp.ndarray, h : jnp.ndarray, bucket : jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
            g_sum = jnp.sum(g[bucket])
            h_sum = jnp.sum(h[bucket])
            return g_sum, h_sum

        G, H = [], []
        for buckets_j in tqdm(self.buckets):    # 处理每个属性j的分桶
            
            G_j, H_j = [], []

            for bucket in buckets_j: # 遍历每个桶，求每个桶内所有元素的一阶梯度和二阶梯度之和
                g_sum, h_sum = self.spu(bucket_sum, num_returns_policy=SPUCompilerNumReturnsPolicy.FROM_USER, user_specified_num_returns=2,static_argnames='bucket')(g, h, bucket)
                G_j.append(g_sum)
                H_j.append(h_sum)
            G.append(G_j)
            H.append(H_j)
        return G, H

    def forward(self, X : FedNdarray) -> SPUObject:
        """
        前向传播，计算每个样本的预测值
        ## Args:
        - X: 输入特征，类型为FedNdarray或SPUObject。在训练阶段，X秘密共享，应为SPUObject；在评估阶段，X纵向划分，应为FedNdarray。
        ## Returns:
        - PYUObject: 每个样本的预测值，类型为PYUObject
        """
        assert isinstance(X, FedNdarray), "X must be either a FedNdarray."
        Quantiles = self.FedQuantiles            
        assert self.company in X.partitions and self.partner in X.partitions, "X must be split by company and partner assigned to this model"
        assert sf.reveal(X.partition_shape()[self.company])[1] == self.split_index, "Share shape mismatch"
        def search_tree(X : FedNdarray | SPUObject, cur : TreeNode | Leaf) -> list[int]:

            def leq(X : jnp.ndarray, j : int, k : int, Quantiles : jnp.ndarray) -> jnp.ndarray:
                """ 判断X[:, j]是否小于等于当前节点的阈值Quantiles[j, k] """
                return X[:, j] <= Quantiles[j, k]
            
            num_samples = X.shape[0]
            if num_samples == 0:
                return []
            if cur.type == 'leaf':
                return [cur.num] * num_samples
            elif cur.type == 'node':

                j, k = cur.threshold
                # 对于Company的特征，交由Company处理；对于Partner的特征，交由Partner处理
                if j < self.split_index:
                    X_c = X.partitions[self.company]
                    Quantiles_c = Quantiles.partitions[self.company]
                    Xj_leq_Qjk = self.company(leq)(X_c, j, k, Quantiles_c)
                else:
                    X_p = X.partitions[self.partner]
                    Quantiles_p = Quantiles.partitions[self.partner]
                    Xj_leq_Qjk = self.partner(leq)(X_p, j - self.split_index, k, Quantiles_p)
                Xj_leq_Qjk = sf.reveal(Xj_leq_Qjk)
                # 将X划分为左（小于阈值）右（大于等于阈值）两部分
                left_indices = np.where(Xj_leq_Qjk)[0]
                right_indices = np.where(~Xj_leq_Qjk)[0]
                X_L = X[left_indices]
                X_R = X[right_indices]
                # 递归搜索左子树和右子树
                left_results = search_tree(X_L, cur.left)
                right_results = search_tree(X_R, cur.right)
                # 将左子树和右子树的结果合并
                overall_results = [-1] * num_samples
                for idx, res in zip(left_indices, left_results):
                    overall_results[idx] = res
                for idx, res in zip(right_indices,right_results):
                    overall_results[idx] = res
                return overall_results

        leaves_ids = search_tree(X, self.root)
        leaves_ids = np.array(leaves_ids)
        # 求预测值
        w = self.spu(lambda w, leaf_id : w[leaf_id].reshape(-1,1),static_argnames='leaves_ids')(self.leaf_weights,leaves_ids)
        # 将预测值发送给标签y的持有者
        return w

class SSXGBoost:
    def __init__(self, devices : dict, n_estimators = 3, lambda_ = 1e-5, max_depth = 3, div = False, mission = 'Classification'):
        """
        初始化SSXGBoost模型
        ## Args:
         - devices : 应包含四个字段，每个字段的值应为SPU或PYU。例如：

           devices = {
            'spu': spu,
            'company': company,
            'partner': partner,
           }
        - n_estimators: 树的数量，默认为5
        - lambda_: l2正则化参数，默认为1e-5
        - max_depth: 树的最大深度，默认为3
        - div: 是否使用除法。如果为True，则使用除法计算叶子权重和信息增益；如果为False，则使用优化算法计算叶子权重和增益最大分裂点。
        """
        self.trees : list[Tree] = []
        self.n_estimators = n_estimators
        self.lambda_ = lambda_
        self.max_depth = max_depth
        self.div = div
        self.devices = devices
        self.spu = devices['spu']
        self.company = devices['company']
        self.partner = devices['partner']
        self.mission = mission

    def _forward(self, X : FedNdarray | SPUObject)-> SPUObject:
        """
        前向传播。将每棵树的预测值相加，得到最终的预测值。
        ## Args:
        - X: 输入特征，类型为FedNdarray或SPUObject。在训练阶段，X秘密共享，应为SPUObject；在评估阶段，X纵向划分，应为FedNdarray。
        ## Returns:
        - PYUObject: 每个样本的预测值，类型为PYUObject
        """
        preds = 0
        for m in self.trees:
            preds = self.spu(lambda x, y : x + y)(preds, m.forward(X))
        return preds

    def predict(self, X : FedNdarray | SPUObject, device : PYU) -> PYUObject:
        """
        预测。将输入特征X传入模型，得到预测标签。
        ## Args:
        - X: 输入特征，类型为FedNdarray。
        ## Returns:
        - PYUObject: 每个样本的预测标签，类型为PYUObject
        """
        y = self._forward(X).to(device)
        y = device(self.activate_fn)(y)
        # 将概率转换为标签
        y = device(to_int_labels)(y)
        return y

    def fit(self, X : SPUObject, y : PYUObject, buckets : np.ndarray, FedQuantiles : FedNdarray, X_test : FedNdarray = None, y_test : PYUObject = None):
        """
        训练SSXGBoost模型
        ## Args:
         - X: 秘密共享的输入特征
         - y: 明文标签， 由label_holder持有
         - buckets: 桶列表（公开）。每个元素bucket_j是特征j的桶列表。bucket_j中的每个桶是一个一维数组，表示桶内元素在X中的索引。
         - FedQuantiles: 分位点列表（纵向划分）
        """
        assert X.device == self.spu, "X must be on SPU of this model."
        self.train_label_keeper = y.device

        self.FedQuantiles = FedQuantiles
        assert isinstance(FedQuantiles, FedNdarray) and self.company in FedQuantiles.partitions and self.partner in FedQuantiles.partitions, "FedQuantiles must be a FedNdarray with partitions for both company and partner."

        num_samples, self.in_features = sf.reveal(self.spu(jnp.shape)(X))
        _, self.out_features = sf.reveal(self.train_label_keeper(jnp.shape)(y))
        self.in_features = int(self.in_features)
        self.out_features = int(self.out_features)

        validate = isinstance(X_test, FedNdarray) and isinstance(y_test, PYUObject)

        y_pred = self.train_label_keeper(jnp.zeros_like)(y)

        if self.mission == 'Regression':
            self.activate_fn = lambda x: x
            loss_fn = mean_square_error
        elif self.mission == 'Classification':
            if self.out_features == 1:
                if self.train_label_keeper == self.spu:
                    self.activate_fn = approx_sigmoid
                else:
                    self.activate_fn = sigmoid
            else:
                assert isinstance(self.train_label_keeper,PYU), "For muiti-class classification, secret-sharing labels not supported"
                self.activate_fn = softmax
            loss_fn = cross_entropy

        train_accs = []
        test_accs = []            
        for i in range(self.n_estimators):
            tree = Tree(self.devices, self.lambda_, self.max_depth, self.div, self.mission)
            tree.fit(X, y, y_pred, buckets, self.FedQuantiles)

            y_t = tree.train_pred.to(self.train_label_keeper)
            y_pred = self.train_label_keeper(lambda x, y: x + y)(y_pred, y_t)
            self.trees.append(tree)
            
            y_pred_train = self.train_label_keeper(self.activate_fn)(y_pred)

            train_loss = self.train_label_keeper(loss_fn)(y, y_pred_train)
            train_loss = sf.reveal(train_loss)

            # 将概率转换为标签
            y_pred_train = self.train_label_keeper(to_int_labels)(y_pred_train)
            train_acc = self.train_label_keeper(compute_accuracy)(y, y_pred_train)
            train_acc = sf.reveal(train_acc)
            train_accs.append(train_acc)
            print(f"==== Iteration {i} ====\nTrain Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
            if validate:
                print("Validating test dataset...")
                y_pred_test = self._forward(X_test).to(y_test.device)
                y_pred_test = y_test.device(self.activate_fn)(y_pred_test)

                test_loss = y_test.device(loss_fn)(y_test, y_pred_test)
                test_loss = sf.reveal(test_loss)

                y_pred_test = y_test.device(to_int_labels)(y_pred_test)

                test_acc = y_test.device(compute_accuracy)(y_test, y_pred_test)
                test_acc = sf.reveal(test_acc)
                test_accs.append(test_acc)
                print(f"Test Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")

        return train_accs, test_accs
    
    def save(self, paths : dict[str, str], ext = 'npy'):
        '''
        ## Args
        - paths: 保存模型的文件夹路径列表，包含company和partner的路径。例如：
        paths = {
            'company': 'path/to/company/model',
            'partner': 'path/to/partner/model'
        }
        '''
        trees = []
        weights1, weights2 = [], []
        for tree in self.trees:
            trees.append(tree.root)
            w1, w2 = self.spu(SS_share, num_returns_policy=SPUCompilerNumReturnsPolicy.FROM_USER, user_specified_num_returns=2)(tree.leaf_weights)
            w1 = w1.to(self.company)
            w2 = w2.to(self.partner)
            weights1.append(w1)
            weights2.append(w2)
        weights1 = self.company(lambda x : jnp.array(x))(weights1)
        weights2 = self.partner(lambda x : jnp.array(x))(weights2)
        info = {
            'in_features': self.in_features,
            'out_features': self.out_features,
            'n_estimators': self.n_estimators,
            'mission': self.mission,
            'max_depth': self.max_depth,
            'div': self.div,
            'lambda_': self.lambda_,
            'save_as': ext
        }
        def save_model(w : np.ndarray, quantiles : np.ndarray, path : str):
            try:
                os.makedirs(path, exist_ok=True)
                print(f"Directory '{path}' created or already exists.")
            except OSError as e:
                print(f"Error creating directory '{path}': {e}")
            
            if ext == 'npy':
                np.save(os.path.join(path, 'weight.npy'), w)
                np.save(os.path.join(path, 'quantiles.npy'), quantiles)
            elif ext == 'csv':
                np.savetxt(os.path.join(path, 'weight.csv'), w, delimiter=',')
                np.savetxt(os.path.join(path, 'quantiles.csv'), quantiles, delimiter=',')
            json.dump(info, open(os.path.join(path, 'info.json'), 'w'))
            with open(os.path.join(path, 'tree.pkl'), 'wb') as f:
                dill.dump(trees, f)
            

        self.company(save_model)(weights1, self.FedQuantiles.partitions[self.company], paths['company'])
        self.partner(save_model)(weights2, self.FedQuantiles.partitions[self.partner], paths['partner'])

    def load(self, paths : dict[str, str]):
        def load_model(path : str):
            info = json.load(open(os.path.join(path, 'info.json'), 'r'))
            ext = info['save_as']
            if ext == 'csv':
                w = np.loadtxt(os.path.join(path, 'weight.csv'), delimiter=',')
                quantiles = np.loadtxt(os.path.join(path, 'quantiles.csv'), delimiter=',')
            else:
                w = np.load(os.path.join(path, 'weight.npy'))
                quantiles = np.load(os.path.join(path, 'quantiles.npy'))
            with open(os.path.join(path, 'tree.pkl'), 'rb') as f:
                trees = dill.load(f)
            return w, trees, quantiles, info

        w1, trees1, quantiles1, info1 = self.company(load_model, num_returns=4)(paths['company'])
        w2, trees2, quantiles2, info2 = self.partner(load_model, num_returns=4)(paths['partner'])
        info1 = sf.reveal(info1)
        info2 = sf.reveal(info2)
        trees1 = sf.reveal(trees1)
        assert info1 == info2, "Model info mismatch"

        self.in_features = info1['in_features']
        self.out_features = info1['out_features']
        self.n_estimators = info1['n_estimators']
        self.mission = info1['mission']
        self.max_depth = info1['max_depth']
        self.div = info1['div']
        self.lambda_ = info1['lambda_']

        self.FedQuantiles = load({self.company: quantiles1, self.partner: quantiles2}, partition_way=PartitionWay.HORIZONTAL)
        self.trees = []
        split_index = sf.reveal(self.FedQuantiles.partition_shape()[self.company])[0]

        if self.mission == 'Regression':
            self.activate_fn = lambda x: x
        elif self.mission == 'Classification':
            if self.out_features == 1:
                    self.activate_fn = sigmoid
            else:
                assert isinstance(self.train_label_keeper,PYU), "For muiti-class classification, secret-sharing labels not supported"
                self.activate_fn = softmax

        for i in range(self.n_estimators):
            t = Tree(self.devices, self.lambda_, self.max_depth, self.div, self.mission)
            t.root = trees1[i]
            weight1 = self.company(lambda x, idx : x[idx])(w1, i).to(self.spu)
            weight2 = self.partner(lambda x, idx : x[idx])(w2, i).to(self.spu)
            t.FedQuantiles = self.FedQuantiles
            t.split_index = split_index
            t.leaf_weights = self.spu(lambda x, y : x+y)(weight1, weight2)
            self.trees.append(t)

import numpy as np
from common import load_dataset
def quantize_buckets(X : np.ndarray, k : int = 50) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ 将每列等频分桶为 k+1 份，并计算 k 个分位点。在PSI之前调用
    ## Args:
    - X: 输入特征矩阵
    - k: 分桶的数量，默认为50
    ## Returns:
    - Quantiles: 分位点列表，形状为 (num_features, k)
    - buckets: 桶列表，每个元素bucket_j是一个特征j的桶列表。每个桶是一个一维数组，表示桶内元素在X中的索引。
    - label_matrix: 标签矩阵，形状与X相同，每个元素表示该样本在对应特征的分桶标签。标签从0到k。
    """
    buckets, Quantiles = [], []

    label_matrix = np.empty(X.shape, dtype=int)

    for j in range(X.shape[1]):
        col = X[:, j]
        # 1) 计算分位点
        qs = np.quantile(col, [(i + 1) / (k + 1) for i in range(k)]).round(3)
        Quantiles_j = []
        # 2) 排序后等分索引
        buckets_j = []
        left = float('-inf')
        right = qs[0]
        for i in range(len(qs)):
            # 计算每个分位点对应的索引范围
            indices = np.where((col > left) & (col <= right))[0]
            if len(indices) > 0:
                indices = indices
                Quantiles_j.append(right)
                buckets_j.append(indices)
                label_matrix[indices, j] = i
            left = right
            right = float('inf') if i == len(qs) - 1 else qs[i + 1]
        # 3) 最后一个分位点对应的索引范围
        indices = np.where(col > left)[0]
        if len(indices) > 0:
            indices = indices
            buckets_j.append(indices)
            label_matrix[indices, j] = k

        Quantiles.append(Quantiles_j)
        buckets.append([g for g in buckets_j])

    Quantiles = np.array(Quantiles)
    buckets = np.array(buckets)
    return Quantiles, buckets, label_matrix

def recover_buckets(label_matrix : np.ndarray) -> np.ndarray:
    """ 将标签矩阵恢复为桶列表。在PSI之后调用，因为PSI执行之后X每个元素经过重新排列，每个元素的索引与PSI之前不同。"""
    buckets = []
    label_matrix = label_matrix.T
    for label_j in label_matrix:
        buckets_j = []
        k = max(label_j)
        for i in range(k+1):
            items_in_buckets = np.where(label_j==i)[0]
            buckets_j.append(items_in_buckets)
        buckets.append(buckets_j)
    return np.array(buckets)

# 直接运行本文件调用这个函数
def SSXGBoost_test(dataset):
    """（不执行PSI）测试XGBoost"""
    train_X, train_y, test_X, test_y = load_dataset(dataset)
    sf.shutdown(barrier_on_shutdown=False)
    from common import MPCInitializer
    mpc_init = MPCInitializer()
    spu = mpc_init.spu
    company = mpc_init.company
    partner = mpc_init.partner

    # 将 train_X 每列等频分桶为 k+1 份，并计算 k 个分位点
    split_index = train_X.shape[1] // 2
    Quantiles1, _, buckets_labels1 = quantize_buckets(train_X[:, :split_index], k=20)
    Quantiles2, _, buckets_labels2 = quantize_buckets(train_X[:, split_index:], k=20)
    buckets = recover_buckets(np.hstack((buckets_labels1, buckets_labels2)))

    Quantiles1 = sf.to(company, Quantiles1)
    Quantiles2 = sf.to(partner, Quantiles2)
    FedQuantiles = load({company: Quantiles1, partner: Quantiles2}, partition_way=PartitionWay.HORIZONTAL)

    model = SSXGBoost(devices={'spu': spu, 'company': company, 'partner': partner}, 
                      max_depth=1, n_estimators=2, div=False)

    # 然后把训练集 secret‐share 到 SPU
    train_X = sf.to(company, np.array(train_X)).to(spu)
    train_y = sf.to(partner, np.array(train_y,dtype=np.float32))

    test_X1, test_X2 = test_X[:, :split_index], test_X[:, split_index:]
    test_X1 = sf.to(company, test_X1)
    test_X2 = sf.to(partner, test_X2)
    test_X = load({company: test_X1, partner: test_X2})
    test_y = sf.to(company, np.array(test_y,dtype=np.float32))

    train_accs, test_accs = model.fit(train_X, train_y, buckets, FedQuantiles,X_test=test_X, y_test=test_y)
    
    import matplotlib.pyplot as plt
    plt.plot(train_accs,label = "train_acc")
    plt.plot(test_accs,label = "test_acc")
    plt.xlabel("nEstimators")
    plt.legend()
    plt.title(f"SSXGBoost_{dataset}")
    plt.savefig(f"SSXGBoost_{dataset}.png")

    model.save({'company':f'SSXGBoost_{dataset}_company','partner':f'SSXGBoost_{dataset}_partner'}, ext='csv')
    model = SSXGBoost(devices={'spu': spu, 'company': company, 'partner': partner})
    model.load({'company':f'SSXGBoost_{dataset}_company','partner':f'SSXGBoost_{dataset}_partner'})
    y_pred = model.predict(test_X, device=company)
    y_pred = sf.reveal(y_pred)
    test_y = sf.reveal(test_y)
    from sklearn.metrics import accuracy_score
    Accuracy = accuracy_score(test_y, y_pred)
    print(f"Accuracy of SSXGBoost on {dataset} dataset after loading: {Accuracy:.4f}")

    # import xgboost as xgb
    # model = xgb.XGBClassifier()

    # if num_cat > 1:
    #     train_y = train_y.argmax(axis=1)

    # model.fit(train_X,train_y.ravel())
    # y_pred = model.predict(test_X)
    # Accuracy = accuracy_score(test_y, y_pred)
    # print(f"Accuracy of XGBoost on {dataset} dataset: {Accuracy:.4f}")
    sf.shutdown()

from time import time
import os
if __name__ == "__main__":
    start_time = time()
    SSXGBoost_test("breast")
    end_time = time()
    print(f"SSXGBoost test completed in {end_time - start_time:.2f} seconds.")
    # SSXGBoost_test("adult")
    os._exit(0)