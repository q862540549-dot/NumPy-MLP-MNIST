# NumPy Implementation of Multi-Layer Perceptron (MLP)

本项目是一个基于 NumPy 手写的深度学习框架，实现了对 MNIST 手写数字集的分类，不依赖 PyTorch/TensorFlow。

## 核心功能
- **全连接层**: 支持前向传播与反向传播。
- **激活函数**: 支持 ReLU, Sigmoid, Tanh, LeakyReLU 及其导数。
- **正则化**: L2 Regularization & Dropout (Inverted Dropout)。
- **优化器**: Adam Optimizer & Mini-batch SGD (with Momentum)。
- **初始化**: He Initialization & Xavier Initialization。

## 实验结果
在 10 个 Epoch 内，Baseline 模型在验证集上的准确率达到了 **97.8%**，超过了实验要求的 95%。

## 如何运行
```bash
conda env create -f environment.yml
conda activate mlp_numpy
python train.py