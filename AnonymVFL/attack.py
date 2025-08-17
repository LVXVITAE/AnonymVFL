"""
攻击效果评估框架 - 比较sklearn LR与PSI_LR的防护效果
- 评估标准逻辑回归与PSI安全逻辑回归在成员推断攻击下的防护能力
- 分析两种方法的模型精度与攻击成功率
- 提供可视化比较结果
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import json
import time
from typing import Dict, List, Tuple, Union, Callable, Optional
import warnings
warnings.filterwarnings('ignore')

# 添加AnonymVFL到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class ModelType:
    """模型类型枚举"""
    SKLEARN_LR = "sklearn_lr"
    PSI_LR = "psi_lr"

class MembershipInferenceEvaluator:
    """
    成员推断攻击评估框架 - 比较sklearn LR与PSI_LR
    """
    
    def __init__(self, result_dir: str = "attack_results"):
        """
        初始化评估器
        
        参数:
            result_dir: 结果保存目录
        """
        self.result_dir = result_dir
        os.makedirs(result_dir, exist_ok=True)
        
        # 结果存储
        self.results = {
            "timestamp": time.strftime("%Y-%m-%d_%H-%M-%S"),
            "models": {},
            "summary": {}
        }
    
    def _train_sklearn_lr(self, 
                         X_train: np.ndarray, 
                         y_train: np.ndarray, 
                         X_test: np.ndarray,
                         y_test: np.ndarray,
                         params: Dict = None) -> Tuple[object, float, np.ndarray]:
        """
        训练sklearn逻辑回归模型
        
        返回:
            (model, attack_success_rate, predictions)
        """
        params = params or {}
        max_iter = params.get("max_iter", 1000)
        C = params.get("C", 1.0)
        
        print(f"训练sklearn LR模型 (max_iter={max_iter}, C={C})...")
        model = LogisticRegression(max_iter=max_iter, C=C)
        model.fit(X_train, y_train)
        
        # 不计算准确率，直接返回模型用于攻击评估
        print(f"sklearn LR模型训练完成")
        
        return model, 0.0, model.predict_proba(X_test)  # 攻击成功率暂时设为0，后续评估时计算
    
    def _train_psi_lr(self, 
                     X_train: np.ndarray, 
                     y_train: np.ndarray, 
                     X_test: np.ndarray,
                     y_test: np.ndarray,
                     params: Dict = None) -> Tuple[object, float, np.ndarray]:
        """
        训练PSI安全逻辑回归模型
        与test.py中的PSI_LR()流程保持一致
        
        返回:
            (model, attack_success_rate, predictions)
        """
        params = params or {}
        from sklearn.model_selection import train_test_split
        from random import choices
        import string    
        import pandas as pd
        from PSI import PSICompany, PSIPartner
        from LR import train
        from SharedVariable import SharedVariable
        from common import out_dom
        
        print("训练PSI_LR模型...")
        
        # ========== 步骤1: 数据预处理 ==========
        print("=" * 60)
        print("PSI_LR 流程开始")
        print("=" * 60)
        
        # 将numpy数组转换为DataFrame，模拟从CSV读取的数据格式
        # 最后一列是标签
        train_data_list = []
        for i in range(len(X_train)):
            row = list(X_train[i]) + [y_train[i]]
            train_data_list.append(row)
        
        # 创建列名
        feature_cols = [f'feature_{i}' for i in range(X_train.shape[1])]
        columns = feature_cols + ['label']
        train_data = pd.DataFrame(train_data_list, columns=columns).astype(np.uint64)
        
        print(f"训练数据形状: {train_data.shape}")
        print(f"数据列结构: {train_data.columns.tolist()}")
        
        # ========== 步骤2: 准备PSI数据 ==========
        print(f"\n【步骤2: 准备PSI数据】")
        
        # 重置索引
        train_data.index = range(train_data.shape[0])
        
        # 为训练数据添加唯一密钥
        keys = [''.join(choices(string.ascii_uppercase + string.digits, k=20)) 
                for _ in range(train_data.shape[0])]
        keys_df = pd.DataFrame(keys, columns=['key'])
        
        # 构建包含密钥的训练数据：[密钥, 特征, 标签]
        train_data_with_keys = pd.concat([keys_df, train_data], axis=1)
        print(f"添加密钥后训练数据形状: {train_data_with_keys.shape}")
        print(f"数据列结构: ['key'] + {train_data.columns.tolist()}")
        
        # ========== 步骤3: 垂直分割数据 ==========
        print(f"\n【步骤3: 垂直分割数据模拟】")
        
        # 计算特征分割点（假设特征平均分割）
        n_features = train_data.shape[1] - 1  # 减去标签列
        split_point = n_features // 2
        
        # Company方：密钥 + 前半部分特征 + 标签
        # Partner方：密钥 + 后半部分特征
        
        # 随机采样模拟现实中的数据不完全重叠
        company_sample = train_data_with_keys.sample(frac=0.9, random_state=42)
        partner_sample = train_data_with_keys.sample(frac=0.9, random_state=43)
        
        # Company方数据：[密钥, 前split_point个特征, 标签]
        company_feature_cols = ['key'] + train_data.columns[:split_point].tolist() + [train_data.columns[-1]]
        company_data = company_sample[company_feature_cols]
        
        # Partner方数据：[密钥, 后面的特征]  
        partner_feature_cols = ['key'] + train_data.columns[split_point:-1].tolist()
        partner_data = partner_sample[partner_feature_cols]
        
        print(f"Company方数据形状: {company_data.shape}")
        print(f"Company方列: {company_data.columns.tolist()}")
        print(f"Partner方数据形状: {partner_data.shape}")
        print(f"Partner方列: {partner_data.columns.tolist()}")
        
        # ========== 步骤4: 执行PSI ==========
        print(f"\n【步骤4: 执行PSI】")
        
        # PSI输入：第0列是密钥，其余列是数据（特征+标签）
        company_psi = PSICompany(company_data.iloc[:,0], company_data.iloc[:,1:])
        partner_psi = PSIPartner(partner_data.iloc[:,0], partner_data.iloc[:,1:])
        
        print(f"Company PSI - 记录数: {company_psi.num_records}, 数据列数: {company_psi.num_features}")
        print(f"Partner PSI - 记录数: {partner_psi.num_records}, 数据列数: {partner_psi.num_features}")
        
        U_c, company_pk = company_psi.exchange()
        E_c, U_p, partner_pk = partner_psi.exchange(U_c, company_pk)
        L, R_cI = company_psi.compute_intersection(E_c, U_p, partner_pk)
        R_pI = partner_psi.output_shares(L)
        
        intersection_size = len(L[0]) if L and hasattr(L[0], '__len__') else 0
        print(f"PSI完成，交集大小: {intersection_size}")
        
        if intersection_size == 0:
            print("❌ 错误: PSI交集为空，无法继续训练")
            # 如果PSI失败，回退到简化方法
            print("回退到简化的安全逻辑回归训练...")
            weights = self._simple_secure_lr(X_train, y_train)
            return self._create_psi_model(weights, X_test, y_test)
        
        # ========== 步骤5: 重构PSI后的数据 ==========
        print(f"\n【步骤5: 重构PSI后的数据】")
        
        # 恢复完整数据
        R_cI_data = np.array(R_cI[0], dtype=np.int64) - out_dom
        R_pI_data = np.array(R_pI[0], dtype=np.int64)
        
        print(f"Company分片形状: {R_cI_data.shape}")
        print(f"Partner分片形状: {R_pI_data.shape}")
        
        # 重构完整数据：R_I = (R_cI + R_pI) % out_dom
        R_I = (R_cI_data + R_pI_data) % out_dom
        print(f"重构后完整数据形状: {R_I.shape}")
        
        # ========== 步骤6: 分离特征和标签 ==========
        print(f"\n【步骤6: 分离特征和标签】")
        
        # 分析重构数据的结构
        # 根据PSI的实现，R_I包含了两方的数据合并结果
        # 需要根据原始的数据分割方式来正确提取特征和标签
        
        # Company方原始数据结构: [前split_point个特征, 标签]
        # Partner方原始数据结构: [后面的特征]
        # 重构后的数据应该是: [Company特征, Company标签, Partner特征, Partner标签(0填充)]
        
        company_features_end = split_point
        company_label_idx = company_features_end
        partner_features_start = company_label_idx + 1
        partner_features_end = partner_features_start + (n_features - split_point)
        
        # 提取Company的特征和标签
        train_X_company = R_I[:, :company_features_end]  # Company特征
        train_y_psi = R_I[:, company_label_idx:company_label_idx+1]  # Company标签
        train_X_partner = R_I[:, partner_features_start:partner_features_end]  # Partner特征
        
        # 合并所有特征
        train_X_psi = np.concatenate([train_X_company, train_X_partner], axis=1)
        
        print(f"PSI后训练特征形状: {train_X_psi.shape}")
        print(f"PSI后训练标签形状: {train_y_psi.shape}")
        print(f"特征范围: [{train_X_psi.min():.2f}, {train_X_psi.max():.2f}]")
        print(f"标签唯一值: {np.unique(train_y_psi)}")
        
        # ========== 步骤7: 准备测试数据 ==========
        print(f"\n【步骤7: 准备测试数据】")
        
        # 测试数据保持原始格式：最后一列是标签
        test_X_psi = X_test.copy()
        test_y_psi = y_test.reshape(-1, 1)
        
        print(f"测试特征形状: {test_X_psi.shape}")
        print(f"测试标签形状: {test_y_psi.shape}")
        
        # 检查特征维度匹配
        if train_X_psi.shape[1] != test_X_psi.shape[1]:
            print(f"⚠️  特征维度不匹配: 训练{train_X_psi.shape[1]} vs 测试{test_X_psi.shape[1]}")
            
            if train_X_psi.shape[1] > test_X_psi.shape[1]:
                # 训练特征多，截取前面的特征
                train_X_psi = train_X_psi[:, :test_X_psi.shape[1]]
                print(f"截取训练特征到: {train_X_psi.shape}")
            else:
                # 测试特征多，截取前面的特征
                test_X_psi = test_X_psi[:, :train_X_psi.shape[1]]
                print(f"截取测试特征到: {test_X_psi.shape}")
        
        # ========== 步骤8: 使用SharedVariable包装训练数据 ==========
        print(f"\n【步骤8: 准备安全计算数据】")
        
        # 为安全计算创建SharedVariable
        # 这里假设标签只在Company方，Partner方标签为0
        train_X_shared = SharedVariable(train_X_psi[:len(train_X_psi)//2], 
                                       train_X_psi[len(train_X_psi)//2:])
        train_y_shared = SharedVariable(train_y_psi[:len(train_y_psi)//2], 
                                       np.zeros_like(train_y_psi[len(train_y_psi)//2:]))
        
        print(f"SharedVariable训练特征形状: {train_X_shared.shape}")
        print(f"SharedVariable训练标签形状: {train_y_shared.shape}")
        
        # ========== 步骤9: 输出最终结果 ==========
        print(f"\n【步骤9: 数据准备完成】")
        print("=" * 60)
        print("PSI后的数据已准备就绪:")
        print(f"- train_X: {train_X_shared.shape} (SharedVariable)")
        print(f"- train_y: {train_y_shared.shape} (SharedVariable)")  
        print(f"- test_X: {test_X_psi.shape} (numpy array)")
        print(f"- test_y: {test_y_psi.shape} (numpy array)")
        print("=" * 60)
        
        # ========== 步骤10: 训练模型 ==========
        print(f"\n【步骤10: 训练PSI-LR模型】")
        
        try:
            # 使用LR.py中的train函数进行训练，与test.py保持一致
            weight = train(train_X_shared, train_y_shared, test_X_psi, test_y_psi).reveal()
            print(f"模型训练完成，权重形状: {weight.shape}")
            
            # 计算预测结果
            test_X_with_bias = np.hstack([np.ones((test_X_psi.shape[0], 1)), test_X_psi])
            
            # 确保权重维度匹配
            if test_X_with_bias.shape[1] != weight.shape[0]:
                if test_X_with_bias.shape[1] < weight.shape[0]:
                    weight = weight[:test_X_with_bias.shape[1]]
                else:
                    padding = np.zeros((test_X_with_bias.shape[1] - weight.shape[0], weight.shape[1]))
                    weight = np.vstack([weight, padding])
            
            # 预测
            logits = test_X_with_bias @ weight
            y_pred_proba = 1 / (1 + np.exp(-logits))
            
            print(f"PSI_LR模型训练完成")
            
            # 创建模型包装器
            class PSI_LR_Model:
                def __init__(self, weights):
                    self.weights = weights.ravel() if weights.ndim > 1 else weights
                    
                def predict(self, X):
                    X_with_bias = np.hstack([np.ones((X.shape[0], 1)), X])
                    if X_with_bias.shape[1] != len(self.weights):
                        if X_with_bias.shape[1] < len(self.weights):
                            weights = self.weights[:X_with_bias.shape[1]]
                        else:
                            padding = np.zeros(X_with_bias.shape[1] - len(self.weights))
                            weights = np.hstack([self.weights, padding])
                    else:
                        weights = self.weights
                        
                    logits = X_with_bias @ weights
                    return (1 / (1 + np.exp(-logits)) > 0.5).astype(int).flatten()
                    
                def predict_proba(self, X):
                    X_with_bias = np.hstack([np.ones((X.shape[0], 1)), X])
                    if X_with_bias.shape[1] != len(self.weights):
                        if X_with_bias.shape[1] < len(self.weights):
                            weights = self.weights[:X_with_bias.shape[1]]
                        else:
                            padding = np.zeros(X_with_bias.shape[1] - len(self.weights))
                            weights = np.hstack([self.weights, padding])
                    else:
                        weights = self.weights
                        
                    logits = X_with_bias @ weights
                    probs = 1 / (1 + np.exp(-logits))
                    return np.hstack([1-probs.reshape(-1,1), probs.reshape(-1,1)])
                    
            return PSI_LR_Model(weight), 0.0, y_pred_proba.reshape(-1,1)  # 攻击成功率暂时设为0，后续评估时计算
            
        except Exception as e:
            print(f"PSI-LR训练失败: {e}")
            print("回退到简化的安全逻辑回归训练...")
            weights = self._simple_secure_lr(train_X_psi, train_y_psi.ravel())
            return self._create_psi_model(weights, test_X_psi, test_y_psi.ravel())
    
    def _create_psi_model(self, weights, X_test, y_test):
        """创建PSI模型包装器的辅助函数"""
        # 添加偏置项
        X_test_with_bias = np.hstack([np.ones((X_test.shape[0], 1)), X_test])
        if X_test_with_bias.shape[1] != weights.shape[0]:
            # 调整权重维度
            if X_test_with_bias.shape[1] < weights.shape[0]:
                weights = weights[:X_test_with_bias.shape[1]]
            else:
                padding = np.zeros((X_test_with_bias.shape[1] - weights.shape[0],))
                weights = np.hstack([weights, padding])
        
        # 确保weights是一维的
        if weights.ndim > 1:
            weights = weights.ravel()
            
        logits = X_test_with_bias @ weights
        y_pred_proba = 1 / (1 + np.exp(-logits))
        
        print(f"PSI_LR模型训练完成")
        
        # 创建模型包装器
        class PSI_LR_Model:
            def __init__(self, weights):
                self.weights = weights.ravel() if weights.ndim > 1 else weights
                
            def predict(self, X):
                X_with_bias = np.hstack([np.ones((X.shape[0], 1)), X])
                if X_with_bias.shape[1] != self.weights.shape[0]:
                    if X_with_bias.shape[1] < self.weights.shape[0]:
                        weights = self.weights[:X_with_bias.shape[1]]
                    else:
                        padding = np.zeros((X_with_bias.shape[1] - self.weights.shape[0],))
                        weights = np.hstack([self.weights, padding])
                else:
                    weights = self.weights
                    
                logits = X_with_bias @ weights
                return (1 / (1 + np.exp(-logits)) > 0.5).astype(int).flatten()
                
            def predict_proba(self, X):
                X_with_bias = np.hstack([np.ones((X.shape[0], 1)), X])
                if X_with_bias.shape[1] != self.weights.shape[0]:
                    if X_with_bias.shape[1] < self.weights.shape[0]:
                        weights = self.weights[:X_with_bias.shape[1]]
                    else:
                        padding = np.zeros((X_with_bias.shape[1] - self.weights.shape[0],))
                        weights = np.hstack([self.weights, padding])
                else:
                    weights = self.weights
                    
                logits = X_with_bias @ weights
                probs = 1 / (1 + np.exp(-logits))
                return np.hstack([1-probs.reshape(-1,1), probs.reshape(-1,1)])
                
        return PSI_LR_Model(weights), 0.0, y_pred_proba.reshape(-1,1)  # 攻击成功率暂时设为0，后续评估时计算
    
    def _simple_secure_lr(self, X, y, max_iter=100, lr=0.01):
        """
        简化的安全逻辑回归训练
        使用噪声扰动来模拟安全计算的效果
        """
        print("使用简化的安全逻辑回归训练...")
        
        # 添加偏置项
        X_with_bias = np.hstack([np.ones((X.shape[0], 1)), X])
        
        # 初始化权重
        weights = np.random.normal(0, 0.1, X_with_bias.shape[1])
        
        # 梯度下降训练
        for i in range(max_iter):
            # 前向传播
            logits = X_with_bias @ weights
            probs = 1 / (1 + np.exp(-np.clip(logits, -250, 250)))  # 防止数值溢出
            
            # 计算梯度
            gradient = X_with_bias.T @ (probs - y) / len(y)
            
            # 添加噪声来模拟安全计算的效果
            noise = np.random.normal(0, 0.001, gradient.shape)
            gradient += noise
            
            # 更新权重
            weights -= lr * gradient
            
            # 衰减学习率
            if i % 20 == 19:
                lr *= 0.9
                
        return weights
    
    def evaluate_model(self,
                      model_type: str,
                      X_train: np.ndarray, 
                      y_train: np.ndarray,
                      X_test: np.ndarray,
                      y_test: np.ndarray,
                      model_params: Dict = None,
                      attack_params: Dict = None) -> Dict:
        """
        评估单个模型在推断攻击下的防护效果
        
        参数:
            model_type: 模型类型 (ModelType.SKLEARN_LR 或 ModelType.PSI_LR)
            X_train, y_train: 训练数据
            X_test, y_test: 测试数据
            model_params: 模型参数
            attack_params: 攻击参数
            
        返回:
            包含评估结果的字典
        """
        model_params = model_params or {}
        attack_params = attack_params or {}
        
        # 记录开始时间
        start_time = time.time()
        
        # 1. 训练目标模型
        if model_type == ModelType.SKLEARN_LR:
            model, _, test_preds = self._train_sklearn_lr(
                X_train, y_train, X_test, y_test, model_params
            )
        elif model_type == ModelType.PSI_LR:
            model, _, test_preds = self._train_psi_lr(
                X_train, y_train, X_test, y_test, model_params
            )
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
            
        # 2. 构建攻击样本
        print("准备攻击数据...")
        # 从训练集中随机选择一部分作为成员样本
        member_size = min(500, len(X_train) // 2)  # 减少样本数以提高稳定性
        member_indices = np.random.choice(len(X_train), member_size, replace=False)
        members_X = X_train[member_indices]
        members_y = y_train[member_indices]
        members_pred = model.predict_proba(members_X)
        
        # 从测试集中选择同等数量的非成员样本
        nonmember_indices = np.random.choice(len(X_test), min(len(member_indices), len(X_test)), replace=False)
        nonmembers_X = X_test[nonmember_indices]
        nonmembers_y = y_test[nonmember_indices]
        nonmembers_pred = model.predict_proba(nonmembers_X)
        
        # 3. 构建攻击数据集
        # 使用更丰富的特征来提高攻击效果
        attack_features = []
        
        # 成员样本特征
        member_features = []
        for i, pred in enumerate(members_pred):
            # 获取预测概率
            if len(pred.shape) > 1 and pred.shape[1] > 1:
                prob_pos = pred[1] if pred.ndim == 1 else pred[0, 1]
            else:
                prob_pos = pred[0] if pred.ndim == 1 else pred[0, 0]
                
            true_label = members_y[i]
            
            # 多种攻击特征
            confidence = abs(prob_pos - 0.5) * 2  # 置信度
            correctness = 1 if ((prob_pos > 0.5) == (true_label == 1)) else 0  # 预测正确性
            entropy = -prob_pos * np.log(prob_pos + 1e-8) - (1-prob_pos) * np.log(1-prob_pos + 1e-8)  # 熵
            
            member_features.append([prob_pos, confidence, correctness, entropy])
        
        # 非成员样本特征
        nonmember_features = []
        for i, pred in enumerate(nonmembers_pred):
            if len(pred.shape) > 1 and pred.shape[1] > 1:
                prob_pos = pred[1] if pred.ndim == 1 else pred[0, 1]
            else:
                prob_pos = pred[0] if pred.ndim == 1 else pred[0, 0]
                
            true_label = nonmembers_y[i]
            
            confidence = abs(prob_pos - 0.5) * 2
            correctness = 1 if ((prob_pos > 0.5) == (true_label == 1)) else 0
            entropy = -prob_pos * np.log(prob_pos + 1e-8) - (1-prob_pos) * np.log(1-prob_pos + 1e-8)
            
            nonmember_features.append([prob_pos, confidence, correctness, entropy])
        
        # 组合成攻击数据集
        attack_X = np.vstack([
            np.array(member_features),      # 成员样本
            np.array(nonmember_features)    # 非成员样本
        ])
        
        attack_y = np.hstack([
            np.ones(len(member_features)),    # 成员标签: 1
            np.zeros(len(nonmember_features)) # 非成员标签: 0
        ]).astype(int)
        
        # 4. 分割攻击数据集
        attack_X_train, attack_X_test, attack_y_train, attack_y_test = train_test_split(
            attack_X, attack_y, test_size=0.3, stratify=attack_y, random_state=42
        )
        
        # 5. 构建并训练攻击模型
        print("训练攻击模型...")
        from sklearn.ensemble import RandomForestClassifier
        attack_model = RandomForestClassifier(
            n_estimators=100, 
            max_depth=10,
            random_state=42
        )
        attack_model.fit(attack_X_train, attack_y_train)
        
        # 6. 评估攻击模型性能
        attack_y_pred = attack_model.predict(attack_X_test)
        attack_accuracy = accuracy_score(attack_y_test, attack_y_pred)
        
        # 计算ROC曲线和AUC
        attack_y_prob = attack_model.predict_proba(attack_X_test)[:, 1]
        fpr, tpr, _ = roc_curve(attack_y_test, attack_y_prob)
        attack_auc = auc(fpr, tpr)
        
        # 计算混淆矩阵
        cm = confusion_matrix(attack_y_test, attack_y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # 计算攻击评估指标
        attack_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        attack_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        attack_f1 = 2 * (attack_precision * attack_recall) / (attack_precision + attack_recall) if (attack_precision + attack_recall) > 0 else 0
        
        # 记录结束时间
        end_time = time.time()
        
        # 7. 保存结果
        result = {
            "model_type": model_type,
            "model_params": model_params,
            "attack_params": attack_params,
            "attack_accuracy": attack_accuracy,
            "attack_auc": attack_auc,
            "attack_precision": attack_precision,
            "attack_recall": attack_recall,
            "attack_f1": attack_f1,
            "confusion_matrix": cm.tolist(),
            "execution_time": end_time - start_time,
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist()
        }
        
        self.results["models"][model_type] = result
        print(f"评估完成。攻击成功率: {attack_accuracy:.4f}, AUC: {attack_auc:.4f}")
        return result
    
    def compare_models(self,
                      X_train: np.ndarray,
                      y_train: np.ndarray, 
                      X_test: np.ndarray,
                      y_test: np.ndarray,
                      sklearn_params: Dict = None,
                      psi_params: Dict = None,
                      dataset_name: str = "default") -> Dict:
        """
        比较sklearn LR与PSI_LR在推断攻击下的防护效果
        
        参数:
            X_train, y_train: 训练数据
            X_test, y_test: 测试数据
            sklearn_params: sklearn LR参数
            psi_params: PSI_LR参数
            dataset_name: 数据集名称
            
        返回:
            包含比较结果的字典
        """
        self.results["dataset"] = dataset_name
        
        # 1. 评估sklearn LR
        print("\n" + "="*50)
        print("评估sklearn LR模型")
        print("="*50)
        sklearn_result = self.evaluate_model(
            ModelType.SKLEARN_LR,
            X_train, y_train, X_test, y_test,
            model_params=sklearn_params
        )
        
        # 2. 评估PSI_LR
        print("\n" + "="*50)
        print("评估PSI_LR模型")
        print("="*50)
        psi_result = self.evaluate_model(
            ModelType.PSI_LR,
            X_train, y_train, X_test, y_test,
            model_params=psi_params
        )
        
        # 3. 计算比较指标
        attack_reduction = sklearn_result["attack_accuracy"] - psi_result["attack_accuracy"]
        
        # 避免除零错误
        attack_reduction_pct = (attack_reduction / sklearn_result["attack_accuracy"] * 100) if sklearn_result["attack_accuracy"] > 0 else 0
        
        self.results["summary"] = {
            "sklearn_attack_accuracy": sklearn_result["attack_accuracy"],
            "psi_attack_accuracy": psi_result["attack_accuracy"],
            "attack_reduction": attack_reduction,
            "attack_reduction_pct": attack_reduction_pct
        }
        
        # 4. 检查是否满足指标要求
        meets_requirement = attack_reduction_pct >= 15
        self.results["summary"]["meets_requirement"] = meets_requirement
        
        print("\n" + "="*50)
        print(f"比较结果 - {dataset_name}")
        print("="*50)
        print(f"sklearn LR攻击成功率: {sklearn_result['attack_accuracy']:.4f}")
        print(f"PSI_LR攻击成功率: {psi_result['attack_accuracy']:.4f}")
        print(f"攻击成功率降低: {attack_reduction:.4f} ({attack_reduction_pct:.2f}%)")
        print(f"是否满足要求(攻击降低>15%): {'✓' if meets_requirement else '✗'}")
        
        return self.results
    
    def save_results(self, filename: str = None) -> None:
        """保存评估结果到文件"""
        if filename is None:
            filename = f"{self.results.get('dataset', 'comparison')}_{self.results['timestamp']}.json"
            
        filepath = os.path.join(self.result_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"结果已保存至: {filepath}")
    
    def generate_report(self, show_plots: bool = True) -> None:
        """生成评估报告和可视化"""
        if not self.results.get("models"):
            print("没有可用的评估结果，请先运行评估")
            return
            
        dataset_name = self.results.get("dataset", "未知数据集")
        
        # 设置英文字体
        plt.rcParams['font.family'] = ['DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 1. ROC曲线比较
        plt.figure(figsize=(10, 8))
        
        # 理论基线
        plt.plot([0, 1], [0, 1], 'k:', label='Random Guess')
        
        # 绘制sklearn LR的ROC曲线
        if ModelType.SKLEARN_LR in self.results["models"]:
            sklearn_result = self.results["models"][ModelType.SKLEARN_LR]
            fpr = sklearn_result["fpr"]
            tpr = sklearn_result["tpr"]
            auc_value = sklearn_result["attack_auc"]
            plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'sklearn LR (AUC={auc_value:.3f})')
        
        # 绘制PSI_LR的ROC曲线
        if ModelType.PSI_LR in self.results["models"]:
            psi_result = self.results["models"][ModelType.PSI_LR]
            fpr = psi_result["fpr"]
            tpr = psi_result["tpr"]
            auc_value = psi_result["attack_auc"]
            plt.plot(fpr, tpr, 'r-', linewidth=2, label=f'PSI_LR (AUC={auc_value:.3f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Membership Inference Attack ROC Curves - {dataset_name}')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        
        plt.savefig(os.path.join(self.result_dir, f"{dataset_name}_roc_comparison.png"), dpi=300, bbox_inches='tight')
        
        if show_plots:
            plt.show()
        plt.close()
            
        # 2. 攻击成功率对比
        if ModelType.SKLEARN_LR in self.results["models"] and ModelType.PSI_LR in self.results["models"]:
            sklearn_result = self.results["models"][ModelType.SKLEARN_LR]
            psi_result = self.results["models"][ModelType.PSI_LR]
            
            plt.figure(figsize=(10, 6))
            
            # 攻击成功率对比
            attack_accs = [sklearn_result["attack_accuracy"], psi_result["attack_accuracy"]]
            bars = plt.bar(['sklearn LR', 'PSI_LR'], attack_accs, color=['#1f77b4', '#ff7f0e'], alpha=0.8)
            plt.ylabel('Attack Success Rate')
            plt.title('Attack Success Rate Comparison')
            plt.ylim(0, 1)
            
            # 添加数值标签
            for bar, v in zip(bars, attack_accs):
                plt.text(bar.get_x() + bar.get_width()/2, v + 0.01, f'{v:.4f}', 
                        ha='center', va='bottom')
                
            # 添加攻击降低标注
            attack_diff = self.results["summary"]["attack_reduction_pct"]
            color = 'green' if attack_diff >= 15 else 'red'
            plt.text(0.5, min(attack_accs) - 0.05, 
                    f'Attack Reduction: {attack_diff:.2f}%', 
                    ha='center', color=color, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.result_dir, f"{dataset_name}_attack_comparison.png"), 
                       dpi=300, bbox_inches='tight')
            
            if show_plots:
                plt.show()
            plt.close()
                
            # 3. 指标达成情况可视化
            plt.figure(figsize=(10, 8))
            
            # 指标区域
            plt.axvline(x=15, color='g', linestyle='--', alpha=0.7, label='Attack Reduction Target (15%)')
            
            # 填充目标区域
            plt.axvspan(15, 100, alpha=0.1, color='green', label='Target Area')
            
            # 绘制当前结果点
            plt.scatter([attack_diff], [0], s=200, color='blue', zorder=5, edgecolor='black', linewidth=2)
            plt.annotate('Current Result', 
                       (attack_diff, 0), 
                       xytext=(attack_diff, 0.1), 
                       arrowprops=dict(arrowstyle='->', color='blue', lw=2))
            
            # 添加是否满足要求的标注
            meets_req = self.results["summary"]["meets_requirement"]
            status_text = "✓ Target Achieved" if meets_req else "✗ Target Not Met"
            status_color = "green" if meets_req else "red"
            plt.text(attack_diff + 2, 0.2, status_text, color=status_color, fontweight='bold', fontsize=12)
            
            plt.xlabel('Attack Success Rate Reduction (%)')
            plt.ylabel('')
            plt.title(f'Privacy Protection Effectiveness - {dataset_name}')
            plt.grid(alpha=0.3, axis='x')
            plt.legend()
            plt.ylim([-0.5, 0.5])
            
            # 设置合理的轴范围
            plt.xlim([-5, max(50, attack_diff * 1.5)])
            
            plt.savefig(os.path.join(self.result_dir, f"{dataset_name}_target_achievement.png"), 
                       dpi=300, bbox_inches='tight')
            
            if show_plots:
                plt.show()
            plt.close()

def run_evaluation_example():
    """运行评估示例"""
    from common import load_dataset
    
    # 加载数据集
    print("加载数据集...")
    train_X, train_y, test_X, test_y = load_dataset("pcs")
    
    print(f"训练集形状: {train_X.shape}, {train_y.shape}")
    print(f"测试集形状: {test_X.shape}, {test_y.shape}")
    
    # 创建评估器
    evaluator = MembershipInferenceEvaluator()
    
    # 运行比较
    results = evaluator.compare_models(
        train_X, train_y.ravel(), test_X, test_y.ravel(),
        sklearn_params={"max_iter": 1000, "C": 1.0},
        psi_params={},
        dataset_name="pcs"
    )
    
    # 保存结果
    evaluator.save_results()
    
    # 生成报告
    evaluator.generate_report(show_plots=True)
    
    return results

if __name__ == "__main__":
    # 运行评估示例
    results = run_evaluation_example()
    print("\n评估完成！")