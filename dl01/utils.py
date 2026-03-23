import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

def load_mnist():
    """加载数据并进行 8:2 划分"""
    print("正在加载 MNIST 数据集...")
    # 如果服务器下载极慢，请联系我更换离线读取代码
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    X = X.astype('float32') / 255.0
    # One-hot 编码
    y = y.astype(int)
    Y_oh = np.eye(10)[y]
    # 严格按照 PDF 要求 8:2 划分
    X_train, X_val, y_train, y_val = train_test_split(X, Y_oh, test_size=0.2, random_state=42)
    return X_train, X_val, y_train, y_val

def get_batches(X, y, batch_size):
    """生成 Mini-batch"""
    indices = np.random.permutation(X.shape[0])
    for i in range(0, X.shape[0], batch_size):
        batch_idx = indices[i:i + batch_size]
        yield X[batch_idx], y[batch_idx]

def calculate_accuracy(logits, labels):
    """计算准确率"""
    preds = np.argmax(logits, axis=1)
    true = np.argmax(labels, axis=1)
    return np.mean(preds == true)

def plot_history(results):
    """绘制参数对比图"""
    plt.figure(figsize=(10, 6))
    for r in results:
        plt.plot(r['history'], label=f"{r['name']} (Acc:{r['acc']:.4f})")
    plt.axhline(y=0.95, color='r', linestyle='--', label='95% Threshold')
    plt.title('Hyperparameter Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Val Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig('comparison_result.png')
    print("\n对比图已保存为 comparison_result.png")
    plt.show()