from time import time
import os
import json
# 导入dill序列化库，用于保存和加载Python对象
import dill
import numpy as np
# 导入SecretFlow隐私计算框架
import secretflow as sf
import jax.numpy as jnp
# 从common模块导入各种激活函数和损失函数
from common import sigmoid, approx_sigmoid, SigmoidCrossEntropy, ApproxSigmoidCrossEntropy, SoftmaxCrossEntropy, MeanSquare, softmax, to_int_labels, cross_entropy, mean_square_error, compute_accuracy, compute_f1_metric, compute_for_metric
# 导入秘密共享机器学习基类
from common import SSML
from secretflow.device.device.spu import SPUCompilerNumReturnsPolicy
from secretflow.device import SPUObject, PYUObject
from secretflow import SPU, PYU
from secretflow.data import FedNdarray
from secretflow.data.ndarray import load, PartitionWay
from tqdm.contrib import tzip
from tqdm import tqdm


class TreeNode:
    """
    树的节点，包括两个左右子树（类型为TreeNode或Leaf）和节点分裂的阈值
    """

    def __init__(self, left, right, threshold: tuple[int, int]):
        """
        ## Args:
        - left: 左子树（类型为TreeNode或Leaf）
        - right: 右子树（类型为TreeNode或Leaf）
        - threshold: 分裂的阈值，类型为tuple[int, int]，对应分位点的索引
        """
        self.left = left
        self.right = right
        self.threshold = threshold
        # 标记节点类型为内部节点
        self.type = 'node'


class Leaf:
    """叶子节点，包含叶子权重的索引，叶子权重以数组的形式保存在Tree类中"""
    def __init__(self, num):
        """
        ## Args:
        - num: 叶子权重的索引，类型为int
        """
        # 保存叶子权重在权重列表中的索引
        self.num = num
        # 标记节点类型为叶子节点
        self.type = 'leaf'


class Tree(SSML):
    def __init__(self, devices: dict, lambda_: float = 1e-5, max_depth: int = 3, div: bool = False, mission='Classification'):
        """
        初始化决策树
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
        - mission: 任务类型，'Classification'或'Regression'，默认为'Classification'
        """
        self.max_depth: int = max_depth
        self.lambda_: float = lambda_
        self.div: bool = div
        # 叶子权重列表，存储每个叶子节点的权重
        self.leaf_weights = []
        self.mission = mission
        super().__init__(devices)

    def fit(self, X: SPUObject, y: PYUObject, y_pred: PYUObject | SPUObject, buckets: np.ndarray, FedQuantiles: FedNdarray):
        """
        训练决策树
        ## Args:
         - X: 秘密共享的输入特征
         - y: 明文标签， 由label_holder持有
         - y_pred: 明文预测标签， 由label_holder持有
         - buckets: 桶列表（公开）。每个元素bucket_j是特征j的桶列表。bucket_j中的每个桶是一个一维数组，表示桶内元素在X中的索引。
        """
        self.train_label_keeper: PYU = y.device
        # 断言输入特征必须在SPU上
        assert X.device == self.spu, "X must be on SPU of the model."
        self.FedQuantiles: FedNdarray = FedQuantiles

        self.buckets = buckets

        # 获取company方特征的分隔索引（前多少列属于company）
        self.split_index = sf.reveal(
            self.FedQuantiles.partition_shape()[self.company])[0]

        # 获取训练样本数量和输入特征维度
        self.num_train_samples, self.in_features = sf.reveal(
            self.spu(jnp.shape)(X))
        # 获取标签的输出维度
        _, self.out_features = sf.reveal(self.train_label_keeper(jnp.shape)(y))
        # 将特征维度和输出维度转换为整数类型
        self.in_features = int(self.in_features)
        self.out_features = int(self.out_features)
        # 生成指示向量。指示向量是一个01向量，1表示该数据属于本树节点

        def generate_indicator(X: jnp.ndarray) -> jnp.ndarray:
            '''生成全1指示向量的函数，表示所有样本都属于当前节点'''
            return jnp.ones((X.shape[0], 1), dtype=int)
        # 在SPU上生成初始指示向量
        s = self.spu(generate_indicator)(X)
        # 根据任务类型选择损失函数
        if self.mission == 'Classification':
            # 二分类任务
            if self.out_features == 1:
                # 根据标签持有方是否为SPU选择损失函数
                if self.train_label_keeper == self.spu:
                    # 使用近似sigmoid交叉熵（SPU上不支持精确sigmoid）
                    loss_fn = ApproxSigmoidCrossEntropy()
                else:
                    # 使用标准sigmoid交叉熵
                    loss_fn = SigmoidCrossEntropy()
            else:
                # 多分类任务使用softmax交叉熵
                loss_fn = SoftmaxCrossEntropy()
        elif self.mission == 'Regression':
            # 回归任务使用均方误差
            loss_fn = MeanSquare()

        # 计算一、二阶梯度（损失函数对预测值的导数）
        g = self.train_label_keeper(loss_fn.grad)(y, y_pred).to(self.spu)
        h = self.train_label_keeper(loss_fn.hess)(y, y_pred).to(self.spu)
        # 初始化训练预测值为0
        self.train_pred = 0.0
        # 构建决策树
        self.root = self._build_tree(g, h, s, 0)
        # 将叶子权重列表转换为NumPy数组
        self.leaf_weights = self.train_label_keeper(lambda x: np.array(x))(self.leaf_weights)
        return

    def __reveal_list(self, arr: list):
        '''DEBUG ONLY'''
        def to_jnp(arr: list):
            arr = jnp.array(arr)
            return arr
        arr = self.spu(to_jnp)(arr)
        return sf.reveal(arr)

    def _leaf(self, g_sum: SPUObject, h_sum: SPUObject, s: SPUObject) -> Leaf:
        """
        创建叶子节点，计算叶子节点的权重。目前只实现了除法版本，尚未实现优化算法版本。
        ## Args:
        - g_sum: 属于本叶子节点的一阶梯度的总和，类型为SPUObject
        - h_sum: 属于本叶子节点的二阶梯度的总和，类型为SPUObject
        ## Returns:
        - Leaf: 叶子节点，包含叶子权重的索引
        """
        # if self.div:
        # 将一、二阶梯度汇总值和指示向量转移到标签持有方
        g_sum = g_sum.to(self.train_label_keeper)
        h_sum = h_sum.to(self.train_label_keeper)
        s = s.to(self.train_label_keeper)
        def leaf_weight_div(g_sum: jnp.ndarray, h_sum: jnp.ndarray, lambda_: float) -> jnp.ndarray:
            '''计算叶子权重的函数，使用除法公式'''
            return -g_sum / (h_sum + lambda_)
        # 在标签持有方计算叶子权重
        weight = self.train_label_keeper(leaf_weight_div)(g_sum, h_sum, self.lambda_)
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

        # 在叶子权重列表中添加新叶子权重
        self.leaf_weights.append(weight)

        def update_pred(pred: jnp.ndarray, weight: jnp.ndarray, s: jnp.ndarray):
            '''新增一个叶节点自动更新预测值'''
            return pred + weight * s
        # 更新训练预测值
        self.train_pred = self.train_label_keeper(update_pred)(self.train_pred, weight, s)
        # 返回新创建的叶子节点
        return Leaf(len(self.leaf_weights) - 1)

    def _build_tree(self, g: SPUObject, h: SPUObject, s: SPUObject, depth: int) -> TreeNode | Leaf:
        """
        递归构建决策树子树
        ## Args:
        - g: 本子树节点的一阶梯度，类型为SPUObject。非本节点的数据其对应一阶梯度为0。
        - h: 本子树节点的二阶梯度，类型为SPUObject。非本节点的数据其对应二阶梯度为0。
        - s: 本子树节点的指示向量，类型为SPUObject。本节点数据在指示向量中用1表示，非本节点的数据用0表示。
        - depth: 当前树的深度，类型为int
        """
        def gh_sum(g: jnp.ndarray, h: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
            """
            本节点一阶二阶梯度之和
            """
            g_sum = jnp.sum(g)
            h_sum = jnp.sum(h)
            return g_sum, h_sum

        # 在SPU上计算梯度总和
        g_sum, h_sum = self.spu(
            gh_sum, num_returns_policy=SPUCompilerNumReturnsPolicy.FROM_USER, user_specified_num_returns=2)(g, h)

        # 如果达到最大深度，创建叶子节点
        if depth >= self.max_depth:
            return self._leaf(g_sum, h_sum, s)

        def loss_fraction(g_sum: jnp.ndarray, h_sum: jnp.ndarray, lambda_: float) -> tuple[jnp.ndarray, jnp.ndarray]:
            """ 计算当前节点的目标损失的分子G^2和分母H + lambda"""
            loss_n = g_sum * g_sum
            loss_d = h_sum + lambda_
            return loss_n, loss_d

        # 在SPU上计算损失分子和分母
        loss_n, loss_d = self.spu(loss_fraction, num_returns_policy=SPUCompilerNumReturnsPolicy.FROM_USER,
                                  user_specified_num_returns=2)(g_sum, h_sum, self.lambda_)

        # 聚合每个桶的梯度信息
        G, H = self._aggregate_bucket(g, h)

        # 计算每个分裂点左右的一阶二阶梯度和
        print("Calculating gradient split info...")
        G_L, G_R, H_L, H_R = [], [], [], []

        def split_info(g_L: jnp.ndarray, h_L: jnp.ndarray, g_k: jnp.ndarray, h_k: jnp.ndarray, g_sum: jnp.ndarray, h_sum: jnp.ndarray, lambda_: float):
            '''计算分裂信息'''
            g_L += g_k
            h_L += h_k
            g_R = g_sum - g_L
            h_R = h_sum - h_L
            return g_L, h_L, g_R, h_R, g_L * g_L, h_L + lambda_, g_R * g_R, h_R + lambda_

        # 遍历每个特征的桶
        for G_j, H_j in tzip(G, H):
            g_L, h_L = 0.0, 0.0
            G_L_j, G_R_j, H_L_j, H_R_j = [], [], [], []
            # 遍历每个桶（除最后一个）
            for g_k, h_k in zip(G_j[:-1], H_j[:-1]):
                # 在SPU上计算分裂信息
                g_L, h_L, g_R, h_R, G_L_j_k, H_L_j_k, G_R_j_k, H_R_j_k = self.spu(
                    split_info,
                    num_returns_policy=SPUCompilerNumReturnsPolicy.FROM_USER,
                    user_specified_num_returns=8
                )(g_L, h_L, g_k, h_k, g_sum, h_sum, self.lambda_)

                # 保存当前分裂点的增益分子分母
                G_L_j.append(G_L_j_k)
                G_R_j.append(G_R_j_k)
                H_L_j.append(H_L_j_k)
                H_R_j.append(H_R_j_k)
            # 保存当前特征的所有分裂点信息
            G_L.append(G_L_j)
            G_R.append(G_R_j)
            H_L.append(H_L_j)
            H_R.append(H_R_j)

        # 计算最优分裂点索引，以及最优分裂点增益的正负
        j, k, sign = self._split(G_L, G_R, H_L, H_R, loss_n, loss_d)
        # 如果增益为正，则进行分裂
        if sign:
            # 阈值是某个分位点。为了方便存储，这里保存分位点的索引
            threshold = (j, k)
            # 计算小于阈值的节点指示向量
            s_L = np.zeros((self.num_train_samples, 1))
            for bucket in self.buckets[j][:k+1]:
                s_L[bucket] = 1.0

            # 大于等于阈值的节点指示向量
            s_R = 1.0 - s_L

            if isinstance(self.train_label_keeper, PYU):
                # 标签持有方是PYU，先转到PYU再转到SPU
                s_L = sf.to(self.train_label_keeper, s_L).to(self.spu)
                s_R = sf.to(self.train_label_keeper, s_R).to(self.spu)
            elif self.train_label_keeper == self.spu:
                # 标签持有方是SPU，先转到company再转到SPU
                s_L = sf.to(self.company, s_L).to(self.spu)
                s_R = sf.to(self.company, s_R).to(self.spu)

            def subtree_args(g: jnp.ndarray, h: jnp.ndarray, s: jnp.ndarray, s_L: jnp.ndarray, s_R: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
                """
                计算左子树和右子树的一阶二阶梯度以及指示向量
                """
                # 将指示向量与当前节点的指示向量相乘
                s_L *= s
                s_R *= s
                # 计算左右子树的一、二阶梯度
                g_L = g * s_L
                h_L = h * s_L
                g_R = g * s_R
                h_R = h * s_R
                return g_L, h_L, s_L, g_R, h_R, s_R

            # 在SPU上计算子树参数
            g_L, h_L, s_L, g_R, h_R, s_R = self.spu(
                subtree_args, num_returns_policy=SPUCompilerNumReturnsPolicy.FROM_USER, user_specified_num_returns=6)(g, h, s, s_L, s_R)
            # 递归构建左右子树
            branch_L = self._build_tree(g_L, h_L, s_L, depth + 1)
            branch_R = self._build_tree(g_R, h_R, s_R, depth + 1)
            # 创建内部节点
            node = TreeNode(left=branch_L, right=branch_R, threshold=threshold)
            return node
        else:
            # 增益为负，创建叶子节点
            return self._leaf(g_sum, h_sum, s)

    def _split(self, G_L: list[list[SPUObject]], G_R: list[list[SPUObject]], H_L: list[list[SPUObject]], H_R: list[list[SPUObject]], loss_n: SPUObject, loss_d: SPUObject) -> tuple[int, int, bool]:
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
        # 生成所有可能的分裂点索引
        indices = [(j, k) for j in range(len(G_L)) for k in range(len(G_L[j]))]
        print("Selecting best split ...")

        def gain(g_L: jnp.ndarray, g_R: jnp.ndarray, h_L: jnp.ndarray, h_R: jnp.ndarray, loss_n: jnp.ndarray, loss_d: jnp.ndarray, lambda_: float) -> jnp.ndarray:
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
        if self.div:  # 使用除法直接计算所有节点的增益（尚未测试）
            def argmax_gain(G_L: jnp.ndarray, G_R: jnp.ndarray, H_L: jnp.ndarray, H_R: jnp.ndarray, loss_n: jnp.ndarray, loss_d: jnp.ndarray, lambda_: float) -> tuple[int, bool]:
                """
                计算增益最大值
                """
                G_L = jnp.array(G_L)
                G_R = jnp.array(G_R)
                H_L = jnp.array(H_L)
                H_R = jnp.array(H_R)
                # 计算所有分裂点的增益
                gain_ = gain(G_L, G_R, H_L, H_R, loss_n, loss_d, lambda_)
                # 找到增益最大的索引
                i = jnp.argmax(gain_)
                # 返回索引和增益是否为正
                return i, gain_.flatten()[i] > 0
            # 在SPU上计算最优分裂点
            i, sign = self.spu(argmax_gain, num_returns_policy=SPUCompilerNumReturnsPolicy.FROM_USER,
                               user_specified_num_returns=2)(G_L, G_R, H_L, H_R, loss_n, loss_d, self.lambda_)
            # 揭示索引和增益符号
            i = sf.reveal(i)
            sign = sf.reveal(sign).item()
            # 获取最优特征索引和分位点索引
            j_opt, k_opt = indices[i]
            return j_opt, k_opt, sign
        else:
            # 不使用除法的比较方法
            def leq(g_L1: list[float], g_R1: list[float], h_L1: list[float], h_R1: list[float], g_L2: list[float], g_R2: list[float], h_L2: list[float], h_R2: list[float]) -> jnp.ndarray:
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

                # 计算比较增益所需的中间变量
                h_L12 = h_L1 * h_L2
                h_R12 = h_R1 * h_R2
                # 计算分子和分母（避免除法）
                nom = h_R12 * (g_L1 * h_L2 - g_L2 * h_L1) + \
                    h_L12 * (g_R1 * h_R2 - g_R2 * h_R1)
                denom = h_L12 * h_R12
                # 通过符号比较判断增益大小
                return (nom > 0) ^ (denom > 0)

            def argmax(G_L: list[list[SPUObject]], G_R: list[list[SPUObject]], H_L: list[list[SPUObject]], H_R: list[list[SPUObject]]) -> tuple[int, int, float, float, float, float]:
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
                # 复制所有候选分裂点
                players = indices.copy()

                # 使用淘汰赛方式找到最优分裂点
                while len(players) > 1:
                    next_round = []
                    # 将候选者分成两组
                    a = players[0::2]
                    b = players[1::2]
                    # 如果候选者数量为奇数，最后一个直接进入下一轮
                    if len(players) % 2 == 1:
                        next_round.append(a[-1])
                        a.pop()

                    # 获取组a的梯度信息
                    g_La = [G_L[i][j] for i, j in a]
                    g_Ra = [G_R[i][j] for i, j in a]
                    h_La = [H_L[i][j] for i, j in a]
                    h_Ra = [H_R[i][j] for i, j in a]

                    # 获取组b的梯度信息
                    g_Lb = [G_L[i][j] for i, j in b]
                    g_Rb = [G_R[i][j] for i, j in b]
                    h_Lb = [H_L[i][j] for i, j in b]
                    h_Rb = [H_R[i][j] for i, j in b]

                    # 在SPU上比较两组的增益
                    a_lt_b = self.spu(leq)(g_La, g_Ra, h_La,
                                           h_Ra, g_Lb, g_Rb, h_Lb, h_Rb)
                    # 揭示比较结果
                    a_lt_b = sf.reveal(a_lt_b)

                    # 根据比较结果选择胜者进入下一轮
                    next_round.extend(
                        [b_i if a_lt_b[i] else a_i for i,
                            (a_i, b_i) in enumerate(zip(a, b))]
                    )

                    # 更新下一轮的候选者
                    players = next_round

                # 获取最终胜者的索引
                j_opt, k_opt = players[0]
                return j_opt, k_opt

            # 执行最优分裂点搜索
            j_opt, k_opt = argmax(G_L, G_R, H_L, H_R)
            # 获取最优分裂点的梯度信息
            g_L_opt = G_L[j_opt][k_opt]
            g_R_opt = G_R[j_opt][k_opt]
            h_L_opt = H_L[j_opt][k_opt]
            h_R_opt = H_R[j_opt][k_opt]

            def max_gain_sign(g_L: jnp.ndarray, g_R: jnp.ndarray, h_L: jnp.ndarray, h_R: jnp.ndarray, loss_n: jnp.ndarray, loss_d: jnp.ndarray, lambda_: float) -> jnp.ndarray:
                """ 计算增益最大分裂点的正负"""
                # 计算中间变量
                h_LR = h_L * h_R
                denom = 2 * h_LR * loss_d
                # 计算增益分子的等价形式
                nom = (g_L * h_R + h_L * g_R - 2 * lambda_ * h_LR) * \
                    loss_d - h_LR * loss_n
                # 返回增益是否为正
                return ~ (nom > 0) ^ (denom > 0)
            # max_gain = self.spu(gain)(g_L_opt, g_R_opt, h_L_opt, h_R_opt, loss_n, loss_d, self.lambda_)
            # max_gain = sf.reveal(max_gain)
            # 在SPU上计算增益符号
            sign = self.spu(max_gain_sign)(g_L_opt, g_R_opt,
                                           h_L_opt, h_R_opt, loss_n, loss_d, self.lambda_)
            # 揭示增益符号
            sign = sf.reveal(sign).item()
            return j_opt, k_opt, sign

    def _aggregate_bucket(self, g: SPUObject, h: SPUObject) -> tuple[list[list[SPUObject]], list[list[SPUObject]]]:
        """
        聚合桶内梯度，求每个桶内的一阶梯度和二阶梯度之和
        ## Args:
        - g: 本节点一阶梯度
        - h: 本节点二阶梯度
        ## Returns:
        - G: 每个桶内的一阶梯度之和列表，和桶列表的形状相同
        - H: 每个桶内的二阶梯度之和列表，和桶列表的形状相同
        """

        print("Aggregating buckets for each feature...")

        def bucket_sum(g: jnp.ndarray, h: jnp.ndarray, bucket: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
            '''计算桶内一、二阶梯度之和'''
            g_sum = jnp.sum(g[bucket])
            h_sum = jnp.sum(h[bucket])
            return g_sum, h_sum

        G, H = [], []
        # 遍历每个特征的桶
        for buckets_j in tqdm(self.buckets):    # 处理每个属性j的分桶

            G_j, H_j = [], []

            # 遍历每个桶，求每个桶内所有元素的一阶梯度和二阶梯度之和
            for bucket in buckets_j:
                # 在SPU上计算桶内梯度和
                g_sum, h_sum = self.spu(bucket_sum, num_returns_policy=SPUCompilerNumReturnsPolicy.FROM_USER,
                                        user_specified_num_returns=2, static_argnames='bucket')(g, h, bucket)
                # 保存当前桶的梯度信息
                G_j.append(g_sum)
                H_j.append(h_sum)
            # 保存当前特征的所有桶梯度信息
            G.append(G_j)
            H.append(H_j)
        return G, H

    def forward(self, X: FedNdarray) -> PYUObject:
        """
        前向传播，计算每个样本的预测值
        ## Args:
        - X: 输入特征，类型为FedNdarray或SPUObject。在训练阶段，X秘密共享，应为SPUObject；在评估阶段，X纵向划分，应为FedNdarray。
        ## Returns:
        - PYUObject: 每个样本的预测值，类型为PYUObject
        """
        # 断言输入必须是联邦数组
        assert isinstance(X, FedNdarray), "X must be either a FedNdarray."
        # 获取分位点数据
        Quantiles = self.FedQuantiles
        # 断言输入必须包含company和partner的分区
        assert self.company in X.partitions and self.partner in X.partitions, "X must be split by company and partner assigned to this model"
        # 验证特征维度是否匹配
        assert sf.reveal(X.partition_shape()[self.company])[
            1] == self.split_index, "Share shape mismatch"

        def search_tree(X: FedNdarray, cur: TreeNode | Leaf) -> list[int]:
            '''递归搜索树'''
            def leq(X: jnp.ndarray, j: int, k: int, Quantiles: jnp.ndarray) -> jnp.ndarray:
                """ 判断X[:, j]是否小于等于当前节点的阈值Quantiles[j, k] """
                return X[:, j] <= Quantiles[j, k]

            # 获取样本数量
            num_samples = X.shape[0]
            # 空输入返回空列表
            if num_samples == 0:
                return []
            # 如果是叶子节点，返回所有样本对应的叶子索引
            if cur.type == 'leaf':
                return [cur.num] * num_samples
            elif cur.type == 'node':

                # 获取分裂阈值
                j, k = cur.threshold
                # 根据特征归属决定由哪一方执行比较
                if j < self.split_index:
                    # 特征属于company
                    X_c = X.partitions[self.company]
                    Quantiles_c = Quantiles.partitions[self.company]
                    Xj_leq_Qjk = self.company(leq)(X_c, j, k, Quantiles_c)
                else:
                    # 特征属于partner
                    X_p = X.partitions[self.partner]
                    Quantiles_p = Quantiles.partitions[self.partner]
                    Xj_leq_Qjk = self.partner(leq)(
                        X_p, j - self.split_index, k, Quantiles_p)
                # 揭示比较结果
                Xj_leq_Qjk = sf.reveal(Xj_leq_Qjk)
                # 根据比较结果划分数据
                left_indices = np.where(Xj_leq_Qjk)[0]
                right_indices = np.where(~Xj_leq_Qjk)[0]
                # 获取左右子树的输入数据
                X_L = X[left_indices]
                X_R = X[right_indices]
                # 递归搜索左右子树
                left_results = search_tree(X_L, cur.left)
                right_results = search_tree(X_R, cur.right)
                # 合并左右子树的结果
                overall_results = [-1] * num_samples
                for idx, res in zip(left_indices, left_results):
                    overall_results[idx] = res
                for idx, res in zip(right_indices, right_results):
                    overall_results[idx] = res
                return overall_results
            else:
                raise ValueError("Invalid tree node type.")

        # 搜索树获取每个样本的叶子索引
        leaves_ids = search_tree(X, self.root)
        # leaves_ids = np.array(leaves_ids)
        # 根据叶子索引获取预测值
        w = self.train_label_keeper(lambda w, leaf_id: w[leaf_id].reshape(-1, 1))(self.leaf_weights, leaves_ids)
        # 将预测值发送给标签y的持有者
        return w


class SSXGBoost(SSML):
    def __init__(self, devices: dict, n_estimators=3, lambda_=1e-5, max_depth=3, div=False, mission='Classification'):
        """
        初始化SSXGBoost模型
        ## Args:
         - devices : 每个字段的值应为SPU或PYU。例如：

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
        self.trees: list[Tree] = []
        self.n_estimators = n_estimators
        self.lambda_ = lambda_
        self.max_depth = max_depth
        self.div = div
        self.devices = devices
        super().__init__(devices)
        self.mission = mission

    def _forward(self, X: FedNdarray) -> PYUObject:
        """
        前向传播。将每棵树的预测值相加，得到最终的预测值。
        ## Args:
        - X: 输入特征，类型为FedNdarray或SPUObject。在训练阶段，X秘密共享，应为SPUObject；在评估阶段，X纵向划分，应为FedNdarray。
        ## Returns:
        - PYUObject: 每个样本的预测值，类型为PYUObject
        """
        # 初始化预测值为0
        preds = 0
        # 累加每棵树的预测值
        for m in self.trees:
            preds = self.train_label_keeper(lambda x, y: x + y)(preds, m.forward(X))
        return preds

    def predict(self, X: FedNdarray, device: PYU) -> PYUObject:
        """
        预测。将输入特征X传入模型，得到预测标签。
        ## Args:
        - X: 输入特征，类型为FedNdarray。
        ## Returns:
        - PYUObject: 每个样本的预测标签，类型为PYUObject
        """
        # 获取预测值并转移到指定设备
        y = self._forward(X).to(device)
        # 应用激活函数
        y = device(self.activate_fn)(y)
        # 将概率转换为标签
        y = device(to_int_labels)(y)
        return y

    def fit(self, X: SPUObject, y: PYUObject, buckets: np.ndarray, FedQuantiles: FedNdarray, X_test: FedNdarray = None, y_test: PYUObject = None):
        """
        训练SSXGBoost模型
        ## Args:
         - X: 秘密共享的输入特征
         - y: 明文标签， 由label_holder持有
         - buckets: 桶列表（公开）。每个元素bucket_j是特征j的桶列表。bucket_j中的每个桶是一个一维数组，表示桶内元素在X中的索引。
         - FedQuantiles: 分位点列表（纵向划分）
        """
        # 断言输入特征必须在SPU上
        assert X.device == self.spu, "X must be on SPU of this model."
        # 保存标签持有方
        self.train_label_keeper = y.device

        # 保存联邦分位点数据
        self.FedQuantiles = FedQuantiles
        # 验证分位点数据的格式
        assert isinstance(FedQuantiles, FedNdarray) and self.company in FedQuantiles.partitions and self.partner in FedQuantiles.partitions, "FedQuantiles must be a FedNdarray with partitions for both company and partner."

        # 获取训练样本数量和特征维度
        num_samples, self.in_features = sf.reveal(self.spu(jnp.shape)(X))
        # 获取标签的输出维度
        _, self.out_features = sf.reveal(self.train_label_keeper(jnp.shape)(y))
        # 将特征维度和输出维度转换为整数
        self.in_features = int(self.in_features)
        self.out_features = int(self.out_features)

        # 判断是否进行验证，如提供了X_test和y_test则进行验证
        validate = isinstance(
            X_test, FedNdarray) and isinstance(y_test, PYUObject)

        # 初始化预测值为零向量
        y_pred = self.train_label_keeper(jnp.zeros_like)(y)

        # 根据任务类型设置激活函数和损失函数
        if self.mission == 'Regression':
            # 回归任务使用恒等激活函数
            self.activate_fn = lambda x: x
            loss_fn = mean_square_error
        elif self.mission == 'Classification':
            # 二分类任务
            if self.out_features == 1:
                # 根据标签持有方类型选择激活函数
                if self.train_label_keeper == self.spu:
                    self.activate_fn = approx_sigmoid
                else:
                    self.activate_fn = sigmoid
            else:
                # 多分类任务使用softmax
                assert isinstance(
                    self.train_label_keeper, PYU), "For muiti-class classification, secret-sharing labels not supported"
                self.activate_fn = softmax
            loss_fn = cross_entropy

        # 初始化性能指标列表
        train_accs = []
        test_accs = []
        train_f1s = []
        train_fors = []
        test_f1s = []
        test_fors = []

        # 初始化最终性能指标
        final_acc = 0.0
        final_f1 = 0.0
        final_for = 1.0

        # 迭代训练每棵树
        for i in range(self.n_estimators):
            # 训练
            tree = Tree(self.devices, self.lambda_,
                        self.max_depth, self.div, self.mission)
            tree.fit(X, y, y_pred, buckets, self.FedQuantiles)

            # 预测
            y_t = tree.train_pred.to(self.train_label_keeper)
            y_pred = self.train_label_keeper(lambda x, y: x + y)(y_pred, y_t)
            self.trees.append(tree)

            # 应用激活函数获取概率
            y_pred_train = self.train_label_keeper(self.activate_fn)(y_pred)

            # 计算训练损失
            train_loss = self.train_label_keeper(loss_fn)(y, y_pred_train)
            train_loss = sf.reveal(train_loss)

            # 将概率转换为标签
            y_pred_train = self.train_label_keeper(to_int_labels)(y_pred_train)
            # 计算训练准确率、F1分数和误漏率
            train_acc = self.train_label_keeper(
                compute_accuracy)(y, y_pred_train)
            train_f1 = self.train_label_keeper(
                compute_f1_metric)(y, y_pred_train)
            train_for = self.train_label_keeper(
                compute_for_metric)(y, y_pred_train)
            train_acc = sf.reveal(train_acc)
            train_f1 = sf.reveal(train_f1)
            train_for = sf.reveal(train_for)
            # 保存训练指标
            train_accs.append(train_acc)
            train_f1s.append(train_f1)
            train_fors.append(train_for)

            print(
                f"Step {i}, Accuracy: {train_acc:.4f}, F1: {train_f1:.4f}, FOR: {train_for:.4f}")
            print(
                f"==== Iteration {i} ====\nTrain Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
            # 如果有测试数据，进行验证
            if validate:
                print("Validating test dataset...")
                # 计算测试集预测值
                y_pred_test = self._forward(X_test).to(y_test.device)
                # 应用激活函数
                y_pred_test = y_test.device(self.activate_fn)(y_pred_test)

                # 计算测试损失
                test_loss = y_test.device(loss_fn)(y_test, y_pred_test)
                test_loss = sf.reveal(test_loss)

                # 将概率转换为标签
                y_pred_test = y_test.device(to_int_labels)(y_pred_test)

                # 计算测试指标
                test_acc = y_test.device(compute_accuracy)(y_test, y_pred_test)
                test_f1 = y_test.device(compute_f1_metric)(y_test, y_pred_test)
                test_for = y_test.device(
                    compute_for_metric)(y_test, y_pred_test)
                test_acc = sf.reveal(test_acc)
                test_f1 = sf.reveal(test_f1)
                test_for = sf.reveal(test_for)
                # 保存测试指标
                test_accs.append(test_acc)
                test_f1s.append(test_f1)
                test_fors.append(test_for)
                print(
                    f"Test Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}, F1: {test_f1:.4f}, FOR: {test_for:.4f}")

        # 选择用于最终评估的指标来源
        metric_source_acc = test_accs if test_accs else train_accs
        metric_source_f1 = test_f1s if test_f1s else train_f1s
        metric_source_for = test_fors if test_fors else train_fors

        # 计算最终指标
        if metric_source_acc:
            final_acc = max(metric_source_acc)
        if metric_source_f1:
            final_f1 = max(metric_source_f1)
        if metric_source_for:
            final_for = min(metric_source_for)

        print(f"\n📈 最终验证结果:")
        print(f"   • 最终准确率: {final_acc:.4f}")
        print(f"   • 最终F1分数: {final_f1:.4f}")
        print(f"   • 最终误漏率: {final_for:.4f}")

        return train_accs, test_accs

    def save(self, paths: dict[str, str], ext='npy'):
        '''
        保存模型
        ## Args
        - paths: 保存模型的文件夹路径列表，包含company和partner的路径。例如：
        paths = {
            'company': 'path/to/company/model',
            'partner': 'path/to/partner/model'
        }
        由于叶子节点权重由梯度计算得到，而梯度是由标签持有方计算的，因此叶子节点权重默认保存在标签持有方的路径下。
        '''
        # 收集所有树的根节点和权重
        trees = []
        weights = []
        for tree in self.trees:
            trees.append(tree.root)
            # 验证叶子权重在正确的设备上
            assert tree.leaf_weights.device == self.train_label_keeper, "Leaf weights must be on train_label_keeper"
            weights.append(tree.leaf_weights)
        # 将权重转换为NumPy数组
        self.train_label_keeper(lambda x: np.array(x))(weights)
        # 创建模型信息字典
        info = {
            'in_features': self.in_features,
            'out_features': self.out_features,
            'n_estimators': self.n_estimators,
            'train_label_keeper': 'company' if self.train_label_keeper == self.company else 'partner' if self.train_label_keeper == self.partner else 'None',
            'mission': self.mission,
            'max_depth': self.max_depth,
            'div': self.div,
            'lambda_': self.lambda_,
            'save_as': ext
        }

        def save_model(quantiles: np.ndarray, path: str, save_ext: str, model_info: dict, tree_roots: list):
            '''保存模型结构'''
            # 创建保存目录
            try:
                os.makedirs(path, exist_ok=True)
                print(f"Directory '{path}' created or already exists.")
            except OSError as e:
                print(f"Error creating directory '{path}': {e}")

            # 根据扩展名保存分位点数据
            if save_ext == 'npy':
                np.save(os.path.join(path, 'quantiles.npy'), quantiles)
            elif save_ext == 'csv':
                # CSV格式不支持不规则数组，保存为npy格式
                np.savetxt(os.path.join(path, 'quantiles.csv'),
                           quantiles, delimiter=',')
            # 保存模型信息为JSON
            json.dump(model_info, open(os.path.join(path, 'info.json'), 'w'))
            # 使用dill保存树结构
            with open(os.path.join(path, 'tree.pkl'), 'wb') as f:
                dill.dump(tree_roots, f)

        def save_weights(weights, save_ext: str, path: str):
            '''保存权重'''
            # 根据扩展名保存权重
            if save_ext == 'npy':
                np.save(os.path.join(path, 'weight.npy'),
                        weights, allow_pickle=True)
            elif save_ext == 'csv':
                np.savetxt(os.path.join(path, 'weight.csv'),
                        weights, delimiter=',')

        # 获取各方的分位点数据
        quantiles1 = self.FedQuantiles.partitions[self.company]
        quantiles2 = self.FedQuantiles.partitions[self.partner]

        # 在各方设备上保存模型
        self.company(save_model)(quantiles1, paths['company'], ext, info, trees)
        self.partner(save_model)(quantiles2, paths['partner'], ext, info, trees)
        # 在标签持有方保存权重
        self.train_label_keeper(save_weights)(weights, ext, paths[info['train_label_keeper']])

    @classmethod
    def load(cls, devices: dict, paths: dict[str, str]) -> 'SSXGBoost':
        """
        加载模型（类方法）
        ## Args: 
         - devices : 每个字段的值应为SPU或PYU。例如：

           devices = {
            'spu': spu,
            'company': company,
            'partner': partner,
           }

         - paths: 加载模型的文件夹路径列表，包含company和partner的路径。例如：
        paths = {
            'company': 'path/to/company/model/dir',
            'partner': 'path/to/partner/model/dir'
        }
        """
        def load_model(path: str):
            '''加载模型'''
            # 读取模型信息
            info = json.load(open(os.path.join(path, 'info.json'), 'r'))
            save_ext = info['save_as']
            # 根据扩展名加载分位点数据
            if save_ext == 'csv':
                quantiles = np.loadtxt(os.path.join(
                    path, 'quantiles.csv'), delimiter=',',ndmin=2)
            else:
                quantiles = np.load(os.path.join(path, 'quantiles.npy'))
            # 使用dill加载树结构
            with open(os.path.join(path, 'tree.pkl'), 'rb') as f:
                trees = dill.load(f)
            return trees, quantiles, info

        # 在各方设备上加载模型
        trees1, quantiles1, info1 = devices['company'](
            load_model, num_returns=3)(paths['company'])
        trees2, quantiles2, info2 = devices['partner'](
            load_model, num_returns=3)(paths['partner'])
        info1 = sf.reveal(info1)
        info2 = sf.reveal(info2)
        trees = sf.reveal(trees1)
        # 验证两方的模型信息一致
        assert info1 == info2, "Model info mismatch"
        info = info1
        # 创建模型实例
        model = cls(devices,
                    n_estimators=info['n_estimators'],
                    lambda_=info['lambda_'],
                    max_depth=info['max_depth'],
                    div=info['div'],
                    mission=info['mission']
                    )
        # 验证设备配置
        assert model.company == devices['company'] and model.partner == devices['partner'], "Company or partner device mismatch"
        # 恢复模型参数
        model.in_features = info['in_features']
        model.out_features = info['out_features']
        model.train_label_keeper = model.company if info['train_label_keeper'] == 'company' else model.partner if info['train_label_keeper'] == 'partner' else None

        def load_weights(path: str, save_ext: str):
            '''加载权重'''
            # 根据扩展名加载权重
            if save_ext == 'csv':
                weights = np.loadtxt(os.path.join(path, 'weight.csv'), delimiter=',',ndmin=2)
            else:
                weights = np.load(os.path.join(path, 'weight.npy'))
            return weights
        # 在标签持有方加载权重
        w = model.train_label_keeper(load_weights)(paths[info['train_label_keeper']], info['save_as'])
        # 重建联邦分位点数据
        model.FedQuantiles = load(
            {model.company: quantiles1, model.partner: quantiles2}, partition_way=PartitionWay.HORIZONTAL)
        # 初始化树列表
        model.trees = []
        # 获取特征分隔索引
        split_index = sf.reveal(
            model.FedQuantiles.partition_shape()[model.company])[0]

        # 根据任务类型设置激活函数
        if model.mission == 'Regression':
            model.activate_fn = lambda x: x
        elif model.mission == 'Classification':
            if model.out_features == 1:
                model.activate_fn = sigmoid
            else:
                assert isinstance(
                    model.train_label_keeper, PYU), "For muiti-class classification, secret-sharing labels not supported"
                model.activate_fn = softmax
        # 重建每棵树
        for i in range(model.n_estimators):
            # 加载树实例
            t = Tree(model.devices, model.lambda_,
                     model.max_depth, model.div, model.mission)
            t.root = trees[i]
            # 恢复树的相关信息
            t.FedQuantiles = model.FedQuantiles
            t.split_index = split_index
            t.leaf_weights = model.train_label_keeper(lambda x, idx: x[idx])(w, i)
            t.train_label_keeper = model.train_label_keeper
            # 将树加入列表
            model.trees.append(t)
        return model

def quantize_buckets(X: np.ndarray, k: int = 50) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ 将每列等频分桶为 k+1 份，并计算 k 个分位点。在PSI之前调用
    ## Args:
    - X: 输入特征矩阵
    - k: 分桶的数量，默认为50
    ## Returns:
    - Quantiles: 分位点列表，形状为 (num_features, k)
    - buckets: 桶列表，每个元素bucket_j是一个特征j的桶列表。每个桶是一个一维数组，表示桶内元素在X中的索引。
    - label_matrix: 标签矩阵，形状与X相同，每个元素表示该样本在对应特征的分桶标签。标签从0到k。
    """
    # 初始化结果列表
    buckets, Quantiles = [], []

    # 创建标签矩阵
    label_matrix = np.empty(X.shape, dtype=int)

    # 遍历每个特征
    for j in range(X.shape[1]):
        # 获取当前特征列
        col = X[:, j]
        # 1) 计算分位点
        qs = np.quantile(col, [(i + 1) / (k + 1) for i in range(k)]).round(3)
        Quantiles_j = []
        # 2) 排序后等分索引
        buckets_j = []
        # 初始化左边界为负无穷
        left = float('-inf')
        # 设置右边界为第一个分位点
        right = qs[0]
        # 遍历每个分位点
        for i in range(len(qs)):
            # 计算每个分位点对应的索引范围
            indices = np.where((col > left) & (col <= right))[0]
            # 如果有样本落入该区间
            if len(indices) > 0:
                indices = indices
                # 保存分位点值
                Quantiles_j.append(right)
                # 保存落入该区间的样本索引
                buckets_j.append(indices)
                # 设置标签
                label_matrix[indices, j] = i
            # 更新左右边界
            left = right
            # 设置下一个右边界
            right = float('inf') if i == len(qs) - 1 else qs[i + 1]
        # 3) 处理最后一个区间的样本
        indices = np.where(col > left)[0]
        # 如果有样本落入最后一个区间
        if len(indices) > 0:
            indices = indices
            # 保存最后一个桶
            buckets_j.append(indices)
            # 设置标签为k
            label_matrix[indices, j] = k

        # 保存当前特征的分位点和桶列表
        Quantiles.append(Quantiles_j)
        buckets.append([g for g in buckets_j])

    # 转换为NumPy数组
    Quantiles = np.array(Quantiles)
    buckets = np.array(buckets)
    return Quantiles, buckets, label_matrix


def recover_buckets(label_matrix: np.ndarray) -> np.ndarray:
    """ 将标签矩阵恢复为桶列表。在PSI之后调用，因为PSI执行之后X每个元素经过重新排列，每个元素的索引与PSI之前不同。"""
    # 初始化桶列表
    buckets = []
    # 转置标签矩阵，按特征遍历
    label_matrix = label_matrix.T
    for label_j in label_matrix:
        buckets_j = []
        # 获取最大标签值（桶数量-1）
        k = max(label_j)
        # 遍历每个桶标签
        for i in range(k+1):
            # 找到属于当前桶的样本索引
            items_in_buckets = np.where(label_j == i)[0]
            buckets_j.append(items_in_buckets)
        buckets.append(buckets_j)
    return np.array(buckets)

# 直接运行本文件调用这个函数


def SSXGBoost_test(dataset):
    """（不执行PSI）测试XGBoost"""
    # 加载数据集
    from common import load_dataset
    train_X, train_y, test_X, test_y = load_dataset(dataset)
    # 关闭现有的SecretFlow集群
    sf.shutdown(barrier_on_shutdown=False)
    # 导入MPC初始化器
    from common import MPCInitializer
    # 创建MPC初始化实例
    mpc_init = MPCInitializer()
    # 获取SPU和各参与方设备
    spu = mpc_init.spu
    company = mpc_init.company
    partner = mpc_init.partner

    # 将 train_X 每列等频分桶为 k+1 份，并计算 k 个分位点
    split_index = train_X.shape[1] // 2
    # 对company和partner的特征进行分桶
    Quantiles1, _, buckets_labels1 = quantize_buckets(
        train_X[:, :split_index], k=20)
    Quantiles2, _, buckets_labels2 = quantize_buckets(
        train_X[:, split_index:], k=20)
    # 合并桶标签并恢复桶列表
    buckets = recover_buckets(np.hstack((buckets_labels1, buckets_labels2)))

    # 将分位点数据转移到各方
    Quantiles1 = sf.to(company, Quantiles1)
    Quantiles2 = sf.to(partner, Quantiles2)
    # 创建联邦分位点数据
    FedQuantiles = load({company: Quantiles1, partner: Quantiles2},
                        partition_way=PartitionWay.HORIZONTAL)

    # 创建SSXGBoost模型
    model = SSXGBoost(devices={'spu': spu, 
                               'company': company, 
                               'partner': partner},
                      max_depth=2, n_estimators=2, div=False)

    # 然后把训练集 secret‐share 到 SPU
    train_X = sf.to(company, np.array(train_X)).to(spu)
    # 将标签转移到partner方
    train_y = sf.to(partner, np.array(train_y, dtype=np.float32))

    # 创建联邦测试特征
    test_X1, test_X2 = test_X[:, :split_index], test_X[:, split_index:]
    test_X1 = sf.to(company, test_X1)
    test_X2 = sf.to(partner, test_X2)
    test_X = load({company: test_X1, partner: test_X2})
    # 将测试标签转移到company方
    test_y = sf.to(company, np.array(test_y, dtype=np.float32))

    # 训练模型
    train_accs, test_accs = model.fit(
        train_X, train_y, buckets, FedQuantiles, X_test=test_X, y_test=test_y)

    import matplotlib.pyplot as plt
    # 绘制训练准确率曲线
    plt.plot(train_accs, label="train_acc")
    # 绘制测试准确率曲线
    plt.plot(test_accs, label="test_acc")
    plt.xlabel("nEstimators")
    plt.legend()
    plt.title(f"SSXGBoost_{dataset}")
    plt.savefig(f"SSXGBoost_{dataset}.png")
    paths = {'company': f'./SSXGBoost_{dataset}_company',
             'partner': f'./SSXGBoost_{dataset}_partner'}
    # 保存模型
    model.save(paths, ext='npy')
    # 加载模型
    model = SSXGBoost.load({'company' : company, 'partner': partner}, paths)
    # 进行预测
    pred_y = model.predict(test_X, device=test_y.device)
    # 计算评分
    scores = model.score(test_y, pred_y)
    print(scores)
    
    # test_X = sf.reveal(test_X)

    # import xgboost as xgb
    # model = xgb.XGBClassifier()

    # if num_cat > 1:
    #     train_y = train_y.argmax(axis=1)

    # model.fit(train_X,train_y.ravel())
    # y_pred = model.predict(test_X)
    # Accuracy = accuracy_score(test_y, y_pred)
    # print(f"Accuracy of XGBoost on {dataset} dataset: {Accuracy:.4f}")
    # 关闭SecretFlow
    sf.shutdown()


if __name__ == "__main__":
    start_time = time()
    SSXGBoost_test("breast")
    end_time = time()
    print(f"SSXGBoost test completed in {end_time - start_time:.2f} seconds.")
    # SSXGBoost_test("adult")
    os._exit(0)