import numpy as np
import matplotlib.pyplot as plt
from model import NeuralNetwork
from utils import load_mnist, get_batches, calculate_accuracy

# 实验组定义
exp_groups = [
    {
        'name': 'Baseline',
        'hidden': [256, 128],
        'act': 'relu',
        'lr': 0.01,
        'batch_size': 64,
        'l2': 1e-4
    },
    {
        'name': 'Group 1',
        'hidden': [512, 256],
        'act': 'leaky_relu',
        'lr': 0.005,
        'batch_size': 128,
        'l2': 1e-3
    },
    {
        'name': 'Group 2',
        'hidden': [128],
        'act': 'tanh',
        'lr': 0.02,
        'batch_size': 32,
        'l2': 0
    }
]

def softmax_ce_loss(Z, Y):
    shift_Z = Z - np.max(Z, axis=1, keepdims=True)
    probs = np.exp(shift_Z) / np.sum(np.exp(shift_Z), axis=1, keepdims=True)
    loss = -np.sum(Y * np.log(probs + 1e-12)) / Y.shape[0]
    return loss, probs - Y

X_train, X_val, y_train, y_val = load_mnist()
results = []

for cfg in exp_groups:
    print(f"\n>>> 正在运行实验: {cfg['name']}")
    model = NeuralNetwork([784] + cfg['hidden'] + [10], act_type=cfg['act'])
    history = []
    
    for epoch in range(10): # 每个实验运行 10 轮
        for xb, yb in get_batches(X_train, y_train, cfg['batch_size']):
            logits = model.forward(xb)
            loss, dZ = softmax_ce_loss(logits, yb)
            model.backward(dZ, cfg['l2'])
            model.update(cfg['lr'])
        
        val_acc = calculate_accuracy(model.forward(X_val), y_val)
        history.append(val_acc)
        print(f"Epoch {epoch+1}: Val Acc = {val_acc:.4f}")
    
    results.append({'name': cfg['name'], 'acc': history[-1], 'history': history})

# 打印最终对比表
print("\n" + "="*50)
print(f"{'实验组':<15} | {'最终验证集准确率':<15}")
print("-"*50)
for r in results:
    print(f"{r['name']:<15} | {r['acc']:<15.4f}")
print("="*50)

# 绘制对比曲线
for r in results:
    plt.plot(r['history'], label=r['name'])
plt.legend()
plt.title('Hyperparameter Comparison')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.savefig('comparison.png')
plt.show()