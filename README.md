# NumPy-MLP-MNIST: HUST 深度学习实验1 (Deep Learning Lab 1)

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/library-NumPy-orange.svg)](https://numpy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## 📌 项目简介 (Project Overview)
本项目为 **HUST 华中科技大学 深度学习课程实验1** 的完整代码实现。

**核心任务**：不使用 PyTorch/TensorFlow 等深度学习框架，仅使用 **NumPy** 实现多层感知机（MLP），完成 MNIST 手写数字分类。本项目通过手动推导反向传播（Backpropagation），实现了包含 **Adam 优化器**、**Dropout 正则化** 及 **He 初始化** 在内的完整工程方案。

---

## 🚀 快速开始 (Quick Start)

按照以下步骤在本地或服务器上复现实验：

## 1. 克隆项目 (Clone the Repository)
打开终端，执行以下命令获取源码：
```bash
git clone https://github.com/q862540549-dot/NumPy-MLP-MNIST.git
cd NumPy-MLP-MNIST
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
