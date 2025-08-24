# AnonymVFL：一个全匿踪纵向联邦学习系统

本项目是一个面向纵向联邦场景的联邦学习系统，旨在解决传统纵向联邦学习中存在的交集ID泄露问题。全匿踪纵向联邦学习分为匿踪求交集和匿踪模型训练两个阶段。系统采用了多种隐私保护技术，包括同态加密、安全多方计算等，确保数据在传输和计算过程中的安全性。

## 1. 开发环境

本项目主要基于[SecretFlow](https://github.com/secretflow/secretflow)框架开发，使用Python语言实现。以下是开发环境的要求：

Python：3.10

pip: >= 19.3

OS: CentOS 7, Ubuntu 20.04

CPU/内存: 建议至少8核16G.

## 2. 安装
```bash
git clone https://github.com/secretflow/secretflow.git
cd AnonymVFL
pip install -r requirements.txt
```

## 3. 模块组成

本系统聚焦于实现匿踪求交集和匿踪逻辑回归以及XGBoost模型训练。核心算法代码在以下四个文件中：
```bash
AnonymVFL/
├── common.py        # 包括数据加载和预处理等公共函数
├── PSI.py            # 匿踪求交集实现
├── LR.py                  # 匿踪逻辑回归实现
└── XGBoost.py              # 匿踪 XGBoost 实现
```

### 3.1 匿踪求交

匿踪求交参考[Private Matching for Compute](https://eprint.iacr.org/2020/599.pdf)论文中的`PS3I`算法。
本文件包含`PSIWorker`、`PSICompany`和`PSIPartner`三个类。`PSIWorker`表示PSI的参与方，主要实现了PSI各参与方的初始化方法。`PSICompany`和`PSIPartner`是`PSIWorker`的子类，分别表示PSI的主从两方。`PSICompany`实现了`exchange`和`compute_intersection`两个方法，分别表示`PS3I`协议中`Company`方密钥数据交换、以及计算交集的步骤。`PSIPartner`实现了`exchange`和`output_shares`两个方法，分别表示`PS3I`协议中`Partner`方密钥数据交换、以及输出共享分片的步骤。该模块提供了一个函数`private_set_intersection`封装了匿踪求交的全过程。算法实现细节详见[源码](AnonymVFL/PSI.py)注释，方法调用说明见[示例](AnonymVFL/PSI_example.ipynb)。

### 3.2 匿踪逻辑回归

匿踪逻辑回归参考[SecureML](https://eprint.iacr.org/2017/396.pdf)论文中的逻辑回归算法，核心思想为使用安全多方计算的方法实现逻辑回归的算子以使用秘密共享数据分片进行模型训练。逻辑回归算法在`SSLR`类中实现，提供`fit`方法进行指定轮数的模型训练，以及`predict`方法进行预测。模型支持L2正则化。算法实现细节详见见[源码](AnonymVFL/LR.py)注释，方法调用说明详见[示例](AnonymVFL/LR_example.ipynb)。

### 3.3 匿踪XGBoost
匿踪XGBoost参考[MP-FedXGB](https://arxiv.org/pdf/2105.05717)中的方法，即秘密共享一阶、二阶梯度信息并使用安全多方计算方法对梯度进行聚合并求信息增益最大分裂点。XGBoost中的梯度提升部分算法在`SSXGBoost`类中实现，该类提供`fit`和`predict`方法，分别用于模型训练和预测。决策树训练部分算法在`Tree`类中实现，该类提供`fit`和`forward`方法，分别用于模型训练和推理。模型支持L2正则化。算法实现细节详见[源码](AnonymVFL/XGBoost.py)注释，方法调用说明详见[示例](AnonymVFL/XGBoost_example.ipynb)。