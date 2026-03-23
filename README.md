# NumPy-MLP-MNIST: HUST 深度学习实验1

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)  
[![NumPy](https://img.shields.io/badge/library-NumPy-orange.svg)](https://numpy.org/)  
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

---

## 📌 项目简介 (Overview)

本项目为 **华中科技大学（HUST）深度学习课程实验一:多层神经网络的numpy实现** 的完整实现。

项目目标是在**不依赖任何深度学习框架（如 PyTorch / TensorFlow）** 的前提下，仅基于 **NumPy** 从零实现一个多层感知机（MLP），完成 **MNIST 手写数字分类任务**。

本项目重点包括：

- 手动推导并实现 **反向传播（Backpropagation）**
- 构建完整训练流程（forward / loss / backward / update）
- 实现常见深度学习组件，提升模型性能与稳定性

---

## ✨ 核心特性 (Key Features)

- ✅ 从零实现 **多层感知机（MLP）**
- ✅ 支持 **ReLU / Softmax** 激活函数
- ✅ 手写 **反向传播（Backpropagation）**
- ✅ 实现 **Adam 优化器**
- ✅ 引入 **Dropout 正则化**
- ✅ 使用 **He 初始化**
- ✅ 模块化设计，易于扩展

---

## 📂 项目结构 (Project Structure)

```bash
NumPy-MLP-MNIST/
│── train.py              # 训练主程序
│── model.py              # MLP模型定义
│── layers.py             # 网络层实现
│── optimizers.py         # 优化器（Adam等）
│── utils.py              # 工具函数
│── environment.yml       # Conda环境
│── README.md             # 项目说明

```
## 🚀 快速开始 (Quick Start)

```
按照以下步骤在本地或服务器上复现实验：

## 1. 克隆项目 (Clone the Repository)
打开终端，执行以下命令获取源码：
```bash
git clone https://github.com/q862540549-dot/NumPy-MLP-MNIST.git
cd NumPy-MLP-MNIST
```
```
## 🛠️ 环境配置 (Environment Setup)
建议使用 Conda 进行环境管理，以确保依赖一致性。

```bash
# 2. 一键创建并激活环境
conda env create -f environment.yml
conda activate mlp_numpy

# 3. 或者手动安装依赖
pip install numpy matplotlib scikit-learn

# 4. 运行与实验
python train.py
```
