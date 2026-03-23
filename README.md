# NumPy-MLP-MNIST: Deep Learning from Scratch

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/library-NumPy-orange.svg)](https://numpy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📌 项目简介 (Project Overview)
本项目是 **HUST 深度学习课程实验1** 的完整实现。在不使用 PyTorch、TensorFlow 等深度学习框架的前提下，仅利用 **NumPy** 库从零构建多层感知机（MLP），实现对 MNIST 手写数字集的分类。

本项目不仅完成了基础的 BP 算法，还集成了 Adam 优化器、Dropout 正则化和 He 初始化等进阶技术，最终在验证集上达到了 **97.8%** 的准确率。

## 🚀 核心特性 (Key Features)
- **纯 NumPy 实现**: 深入底层，手动推导前向传播与反向传播（Backpropagation）。
- **多种激活函数**: 支持 ReLU, LeakyReLU, Tanh, Sigmoid 及其导数实现。
- **进阶优化器**: 实现了 **Adam Optimizer** 与带动量的 SGD，显著加速收敛。
- **正则化技术**: 包含 **L2 Regularization** 与 **Inverted Dropout** 以防止过拟合。
- **权重初始化**: 支持 **He Initialization** 与 **Xavier Initialization**，解决梯度消失问题。
- **自动化实验**: 支持一键运行多组参数调优对比（Baseline vs Group 1 vs Group 2）。

## 🛠️ 环境配置 (Environment Setup)
建议使用 Conda 进行环境管理，以确保依赖一致性。

```bash
# 1. 一键创建并激活环境
conda env create -f environment.yml
conda activate mlp_numpy

# 2. 或者手动安装依赖
pip install numpy matplotlib scikit-learn

# 3. 运行与实验
python train.py
