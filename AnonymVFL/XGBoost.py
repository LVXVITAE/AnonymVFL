# 目前xgboost仅可以跑通，部分算法细节可能还有一些错误和不完善的地方
from ast import Not
import secretflow as sf
import jax.numpy as jnp
from common import sigmoid, SigmoidCrossEntropy, SoftmaxCrossEntropy, softmax
from secretflow.device.device.spu import SPUCompilerNumReturnsPolicy
from secretflow.device.device import SPUObject, PYUObject
from secretflow.data import FedNdarray
from secretflow.data.ndarray import load, PartitionWay
from tqdm import tqdm
from common import MPCInitializer

mpc_init = MPCInitializer()
company, partner, coordinator, spu = mpc_init.company, mpc_init.partner, mpc_init.coordinator, mpc_init.spu
label_holder = company
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

class Leaf:
    """叶子节点，包含叶子权重的索引，叶子权重以数组的形式保存在Tree类中"""
    def __init__(self, num):
        """
        ## Args:
        - num: 叶子权重的索引，类型为int
        """
        self.num = num

class Tree:
    def __init__(self, in_features : int, split_index : int, out_features : int = 1, lambda_ : float = 1e-5, max_depth : int = 3, div : bool =False):
        """
        初始化树
        ## Args:
        - in_features: 输入特征的数量
        - split_index: 表示纵向联邦下划分Company特征和Partner特征的序号。也就是说，X[:, :split_index]是Company的特征，X[:, split_index:]是Partner的特征。
        - out_features: 输出特征的数量，默认为1。目前尚未实现多分类。
        - lambda_: l2正则化参数，默认为1e-5
        - max_depth: 树的最大深度，默认为3
        - div: 是否使用除法。如果为True，则使用除法计算叶子权重和信息增益；如果为False，则使用优化算法计算叶子权重和增益最大分裂点。
        """
        self.max_depth : int = max_depth
        self.in_features : int = in_features
        self.out_features : int = out_features
        if out_features > 1:
            raise NotImplementedError("Multi-class classification is not supported.")
        self.lambda_ : float = lambda_
        self.div : bool = div
        self.split_index = split_index
        # 叶子权重列表，存储每个叶子节点的权重
        self.leaf_weights : list[SPUObject] = []

    def fit(self, X : SPUObject, y : PYUObject, y_pred : PYUObject, buckets : list[list[jnp.ndarray]], SSQuantiles : SPUObject, FedQuantiles : FedNdarray):
        """
        ## Args:
         - X: 秘密共享的输入特征
         - y: 明文标签， 由label_holder持有
         - y_pred: 明文预测标签， 由label_holder持有
         - buckets: 桶列表（公开）。每个元素bucket_j是特征j的桶列表。bucket_j中的每个桶是一个一维数组，表示桶内元素在X中的索引。
        """
        self.SSQuantiles = SSQuantiles
        self.FedQuantiles = FedQuantiles
        loss_fn = SigmoidCrossEntropy if self.out_features == 1 else SoftmaxCrossEntropy
        self.buckets = buckets
        
        # 生成指示向量。指示向量是一个01向量，1表示该数据属于本树节点
        def generate_indicator(X : jnp.ndarray) -> jnp.ndarray:
            return jnp.ones((X.shape[0], 1), dtype=int)
        s = spu(generate_indicator)(X)
        #计算一阶二阶梯度
        def grad_hessian(y : jnp.ndarray, y_pred : jnp.ndarray, loss_fn : SigmoidCrossEntropy | SoftmaxCrossEntropy) -> tuple[jnp.ndarray, jnp.ndarray]:
            g = loss_fn.grad(y, y_pred)
            h = loss_fn.hess(y, y_pred)
            return g, h
        g, h = label_holder(grad_hessian, num_returns=2)(y, y_pred, loss_fn)
        g = g.to(spu)
        h = h.to(spu)
        self.root = self.build_tree(g, h, s, 0)
    def leaf(self, g_sum : SPUObject, h_sum : SPUObject) -> Leaf:
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
        self.leaf_weights.append(weight)    # 在叶子权重列表中添加新叶子权重
        return Leaf(len(self.leaf_weights) - 1)
    def build_tree(self, g : SPUObject, h : SPUObject, s : SPUObject, depth : int) -> TreeNode | Leaf:
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
        
        g_sum, h_sum = spu(gh_sum,num_returns_policy=SPUCompilerNumReturnsPolicy.FROM_USER, user_specified_num_returns=2)(g, h)

        if depth >= self.max_depth:
            return self.leaf(g_sum, h_sum)
        
        def loss_fraction(g_sum : jnp.ndarray, h_sum : jnp.ndarray, lambda_ : float) -> tuple[jnp.ndarray, jnp.ndarray]:
            """ 计算当前节点的目标损失的分子和分母"""
            loss_n = g_sum * g_sum
            loss_d = h_sum + lambda_
            return loss_n, loss_d

        loss_n, loss_d = spu(loss_fraction, num_returns_policy=SPUCompilerNumReturnsPolicy.FROM_USER, user_specified_num_returns=2)(g_sum, h_sum, self.lambda_)

        G, H = self.aggregate_bucket(g, h)
        
        # 计算每个分裂点左右的一阶二阶梯度和
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

        # 计算最优分裂点索引，以及最优分裂点增益的正负
        j, k, sign = self.split(G_L, G_R, H_L, H_R, loss_n, loss_d)
        if sign:
            # 阈值是某个分位点。为了方便存储，这里保存分位点的索引
            threshold = (j, k)
            # 小于阈值的节点指示向量
            s_L = spu(jnp.zeros_like)(s)
            s_L = sf.reveal(s_L).astype(bool)
            for bucket in self.buckets[j][:k]:
                s_L |= jnp.zeros_like(s_L).at[bucket].set(True)
            
            s_L = jnp.astype(s_L, jnp.int32)
            # 大于等于阈值的节点指示向量
            s_R = 1 - s_L
            s_L = sf.to(label_holder, s_L).to(spu)
            s_R = sf.to(label_holder, s_R).to(spu)

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

            g_L, h_L, s_L, g_R, h_R, s_R = spu(subtree_args, num_returns_policy=SPUCompilerNumReturnsPolicy.FROM_USER, user_specified_num_returns=6)(g, h, s, s_L, s_R)
            branch_L = self.build_tree(g_L, h_L, s_L, depth + 1)
            branch_R = self.build_tree(g_R, h_R, s_R, depth + 1)
            node = TreeNode(left=branch_L, right=branch_R, threshold=threshold)
            return node
        else:
            return self.leaf(g_sum, h_sum)

    def split(self, G_L : list[list[SPUObject]], G_R : list[list[SPUObject]], H_L : list[list[SPUObject]], H_R : list[list[SPUObject]], loss_n : SPUObject, loss_d : SPUObject) -> tuple[int, int, bool]:
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
            # 将所有列表摊平
            G_L_flat = spu(jnp.array)([g for G_j in G_L for g in G_j])
            G_R_flat = spu(jnp.array)([g for G_j in G_R for g in G_j])
            H_L_flat = spu(jnp.array)([h for H_j in H_L for h in H_j])
            H_R_flat = spu(jnp.array)([h for H_j in H_R for h in H_j])
            def argmax_gain(G_L_flat : jnp.ndarray, G_R_flat : jnp.ndarray, H_L_flat : jnp.ndarray, H_R_flat : jnp.ndarray, loss_n : jnp.ndarray, loss_d : jnp.ndarray, lambda_ : float) -> tuple[int, bool]:
                """
                计算增益最大值
                """
                gain_ =  gain(G_L_flat, G_R_flat, H_L_flat, H_R_flat, loss_n, loss_d, lambda_)
                i = jnp.argmax(gain_)
                return i, gain_ > 0
            i, sign = spu(argmax_gain)(G_L_flat, G_R_flat, H_L_flat, H_R_flat, loss_n, loss_d, self.lambda_)
            i = sf.reveal(i)
            sign = sf.reveal(sign)
            j_opt, k_opt = indices[i]
            return j_opt, k_opt, sign
        else:
            def gt(g_L1 : list[float], g_R1 : list[float], h_L1 : list[float], h_R1 : list[float], g_L2 : list[float], g_R2 : list[float], h_L2 : list[float], h_R2 : list[float]) -> jnp.ndarray:
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
                """ 计算增益最大分裂点的正负"""
                # h_LR = h_L * h_R
                # denom = h_LR * loss_n
                # nom = g_L * h_R * loss_d + g_R * h_L * loss_d - loss_n * h_LR - lambda_ * denom
                return gain(g_L, g_R, h_L, h_R, loss_n, loss_d, lambda_) > 0
            sign = spu(max_sign)(g_L_opt, g_R_opt, h_L_opt, h_R_opt, loss_n, loss_d, self.lambda_)
            sign = sf.reveal(sign).item()
            return j_opt, k_opt, sign

    def aggregate_bucket(self, g : SPUObject, h : SPUObject) -> tuple[list[list[SPUObject]], list[list[SPUObject]]]:
        """
        求每个桶内的一阶梯度和二阶梯度之和
        ## Args:
        - g: 本节点一阶梯度
        - h: 本节点二阶梯度
        ## Returns:
        - G: 每个桶内的一阶梯度之和列表，和桶列表的形状相同
        - H: 每个桶内的二阶梯度之和列表，和桶列表的形状相同
        """
        
        print("Aggregating buckets...")
        G, H = [], []

        for j, buckets_j in enumerate(self.buckets):    # 处理每个属性j的分桶
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
            for bucket in tqdm(buckets_j): # 遍历每个桶，求每个桶内所有元素的一阶梯度和二阶梯度之和
                g_sum, h_sum = spu(bucket_sum, num_returns_policy=SPUCompilerNumReturnsPolicy.FROM_USER, user_specified_num_returns=2)(g, h, bucket)
                G_j.append(g_sum)
                H_j.append(h_sum)
            G.append(G_j)
            H.append(H_j)
        return G, H

    def forward(self, X : FedNdarray | SPUObject) -> PYUObject:
        """
        前向传播，计算每个样本的预测值
        ## Args:
        - X: 输入特征，类型为FedNdarray或SPUObject。在训练阶段，X秘密共享，应为SPUObject；在评估阶段，X纵向划分，应为FedNdarray。
        ## Returns:
        - PYUObject: 每个样本的预测值，类型为PYUObject
        """
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
                """ 判断X[:, j]是否小于等于当前节点的阈值Quantiles[j, k] """
                return X[:, j] <= Quantiles[j, k]
            
            if mode == 'eval':
                
                num_samples = X.shape[0]
                if num_samples == 0:
                    return []
                if isinstance(cur, Leaf):
                    return [cur.num] * num_samples
                elif isinstance(cur, TreeNode):
                    
                    j, k = cur.threshold
                    # 对于Company的特征，交由Company处理；对于Partner的特征，交由Partner处理
                    if j < self.split_index:
                        X_c = X.partitions[company]
                        Quantiles_c = Quantiles.partitions[company]
                        X_leq_Qjk = company(leq)(X_c, j, k, Quantiles_c)
                    else:
                        X_p = X.partitions[partner]
                        Quantiles_p = Quantiles.partitions[partner]
                        X_leq_Qjk = partner(leq)(X_p, j - self.split_index, k, Quantiles_p)
                    X_leq_Qjk = sf.reveal(X_leq_Qjk)
                    # 将X划分为左（小于阈值）右（大于等于阈值）两部分
                    left_indices = jnp.where(X_leq_Qjk)[0]
                    right_indices = jnp.where(~X_leq_Qjk)[0]
                    X_L = X[left_indices]
                    X_R = X[right_indices]
                    # 递归搜索左子树和右子树
                    left_results = search_tree(X_L, cur.left)
                    right_results = search_tree(X_R, cur.right)
                    # 将左子树和右子树的结果合并
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
                    # 将X划分为左（小于阈值）右（大于等于阈值）两部分
                    left_indices = jnp.where(X_leq_Qjk)[0]
                    right_indices = jnp.where(~X_leq_Qjk)[0]
                    def X_LR(X : jnp.ndarray, left_indices : jnp.ndarray, right_indices : jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
                        return X[left_indices], X[right_indices]
                    X_L, X_R = spu(X_LR, num_returns_policy=SPUCompilerNumReturnsPolicy.FROM_USER, user_specified_num_returns=2)(X, left_indices, right_indices)
                    # 递归搜索左子树和右子树
                    left_results = search_tree(X_L, cur.left)
                    right_results = search_tree(X_R, cur.right)
                    # 将左子树和右子树的结果合并
                    overall_results = [-1] * num_samples
                    for i, idx in enumerate(left_indices):
                        overall_results[idx] = left_results[i]
                    for i, idx in enumerate(right_indices):
                        overall_results[idx] = right_results[i]
                    return overall_results
            else:
                raise ValueError("Mode must be 'train' or 'eval'.")
            
        leaves_ids = search_tree(X, self.root)
        # 求预测值
        w = [self.leaf_weights[leaf_id] for leaf_id in leaves_ids]
        def ret(w : list[jnp.ndarray]):
            return jnp.array(w, dtype=jnp.float32).reshape(-1,1)
        # 将预测值发送给标签y的持有者
        return spu(ret)(w).to(label_holder)

class SSXGBoost:
    def __init__(self,in_features, split_index : int | list[int], out_features = 1, n_estimators = 5, lambda_ = 1e-5, max_depth = 3, div = False):
        """
        初始化SSXGBoost模型
        ## Args:
        - in_features: 输入特征的数量
        - split_index: 表示纵向联邦下划分Company特征和Partner特征的序号。也就是说，X[:, :split_index]是Company的特征
        - out_features: 输出特征的数量，默认为1。目前尚未实现多分类。
        - n_estimators: 树的数量，默认为5
        - lambda_: l2正则化参数，默认为1e-5
        - max_depth: 树的最大深度，默认为3
        - div: 是否使用除法。如果为True，则使用除法计算叶子权重和信息增益；如果为False，则使用优化算法计算叶子权重和增益最大分裂点。
        """
        self.trees : list[Tree] = []
        self.in_features = in_features
        self.out_features = out_features
        self.n_estimators = n_estimators
        self.lambda_ = lambda_
        self.max_depth = max_depth
        self.div = div
        self.split_index = split_index 

    def forward(self, X : FedNdarray | SPUObject):
        """
        前向传播。将每棵树的预测值相加，得到最终的预测值。
        ## Args:
        - X: 输入特征，类型为FedNdarray或SPUObject。在训练阶段，X秘密共享，应为SPUObject；在评估阶段，X纵向划分，应为FedNdarray。
        ## Returns:
        - PYUObject: 每个样本的预测值，类型为PYUObject
        """
        preds = 0
        for m in self.trees:
            preds = label_holder(jnp.add)(preds, m.forward(X))
        return preds

    def predict(self, X : FedNdarray):
        """
        预测。将输入特征X传入模型，得到预测标签。
        ## Args:
        - X: 输入特征，类型为FedNdarray。
        ## Returns:
        - PYUObject: 每个样本的预测标签，类型为PYUObject
        """
        y = self.forward(X)
        y = sf.reveal(y)
        y = sigmoid(y) if self.out_features == 1 else softmax(y) # 将预测值转换为概率

        # 将概率转换为标签
        if self.out_features == 1:
            return y.round()
        else:
            return y.argmax(axis=1).reshape(-1,1)

    def fit(self, X : SPUObject, y : PYUObject, buckets : list[list[jnp.ndarray]], SSQuantiles : SPUObject, FedQuantiles : FedNdarray):
        """
        训练SSXGBoost模型
        ## Args:
         - X: 秘密共享的输入特征
         - y: 明文标签， 由label_holder持有
         - buckets: 桶列表（公开）。每个元素bucket_j是特征j的桶列表。bucket_j中的每个桶是一个一维数组，表示桶内元素在X中的索引。
         - SSQuantiles: 分位点列表（秘密共享）
         - FedQuantiles: 分位点列表（纵向划分）
        """
        self.SSQuantiles = SSQuantiles
        self.FedQuantiles = FedQuantiles

        y_pred = label_holder(jnp.zeros_like)(y)
        for _ in range(self.n_estimators):
            tree = Tree(self.in_features, self.split_index, self.out_features, self.lambda_, self.max_depth, self.div)
            tree.fit(X, y, y_pred, buckets, self.SSQuantiles, self.FedQuantiles)

            y_t = tree.forward(X)
            y_pred = label_holder(jnp.add)(y_pred, y_t)
            self.trees.append(tree)

import numpy as np
from sklearn.metrics import accuracy_score
from common import load_dataset
def quantize_buckets(X : np.ndarray, k : int = 50) -> tuple[np.ndarray, list[list[jnp.ndarray]], np.ndarray]:
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

    label_matrix = np.zeros(X.shape, dtype=int)

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
            indices = jnp.where((col > left) & (col <= right))[0]
            if len(indices) > 0:
                indices = indices.sort()
                Quantiles_j.append(right)
                buckets_j.append(indices)
                label_matrix[indices, j] = i
            left = right
            right = float('inf') if i == len(qs) - 1 else qs[i + 1]
        # 3) 最后一个分位点对应的索引范围
        indices = jnp.where(col > left)[0]
        if len(indices) > 0:
            indices = indices.sort()
            buckets_j.append(indices)
            label_matrix[indices, j] = k

        Quantiles.append(Quantiles_j)
        buckets.append([g for g in buckets_j])

    Quantiles = np.array(Quantiles)
    return Quantiles, buckets, label_matrix

def recover_buckets(label_matrix : np.ndarray) -> list[list[jnp.ndarray]]:
    """ 将标签矩阵恢复为桶列表。在PSI之后调用，因为PSI执行之后X每个元素经过重新排列，每个元素的索引与PSI之前不同。"""
    buckets = []
    label_matrix = label_matrix.T
    for label_j in label_matrix:
        buckets_j = []
        k = max(label_j)
        for i in range(k+1):
            items_in_buckets = jnp.where(label_j==k)[0]
            buckets_j.append(items_in_buckets)
        buckets.append(buckets_j)
    return buckets

# 直接运行本文件调用这个函数
def SSXGBoost_test(dataset):
    """（不执行PSI）测试XGBoost"""
    train_X, train_y, test_X, test_y = load_dataset(dataset)


    # 将 train_X 每列等频分桶为 k+1 份，并计算 k 个分位点
    split_index = train_X.shape[1] // 2
    Quantiles1, _, buckets_labels1 = quantize_buckets(train_X[:, :split_index], k=50)
    Quantiles2, _, buckets_labels2 = quantize_buckets(train_X[:, split_index:], k=50)
    buckets1 = recover_buckets(buckets_labels1)
    buckets2 = recover_buckets(buckets_labels2)
    np.save("Quantiles1.npy", Quantiles1)
    np.save("Quantiles2.npy", Quantiles2)
    Quantiles = np.concatenate((Quantiles1, Quantiles2), axis=0)
    buckets = buckets1 + buckets2
    SSQuantiles = sf.to(label_holder, jnp.array(Quantiles)).to(spu)
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

from time import time
if __name__ == "__main__":
    start_time = time()
    SSXGBoost_test("breast")
    end_time = time()
    print(f"SSXGBoost test completed in {end_time - start_time:.2f} seconds.")
    # SSXGBoost_test("adult")