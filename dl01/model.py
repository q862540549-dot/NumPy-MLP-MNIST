import numpy as np

class FullyConnectedLayer:
    def __init__(self, in_dim, out_dim, init_type='he'):
        # 自动选择初始化方式
        if init_type == 'he':
            self.W = np.random.randn(in_dim, out_dim) * np.sqrt(2.0 / in_dim)
        else: # Xavier
            self.W = np.random.randn(in_dim, out_dim) * np.sqrt(1.0 / (in_dim + out_dim))
        self.b = np.zeros((1, out_dim))
        self.vW, self.vb = np.zeros_like(self.W), np.zeros_like(self.b) # 用于动量

    def forward(self, X):
        self.X = X
        return np.dot(X, self.W) + self.b

    def backward(self, dZ, l2_lambda):
        m = self.X.shape[0]
        self.dW = (np.dot(self.X.T, dZ) / m) + l2_lambda * self.W
        self.db = np.sum(dZ, axis=0, keepdims=True) / m
        return np.dot(dZ, self.W.T)

class Activation:
    def __init__(self, mode='relu'):
        self.mode = mode

    def forward(self, Z):
        self.Z = Z
        if self.mode == 'relu': return np.maximum(0, Z)
        if self.mode == 'leaky_relu': return np.where(Z > 0, Z, 0.01 * Z)
        if self.mode == 'tanh': return np.tanh(Z)
        if self.mode == 'sigmoid': return 1 / (1 + np.exp(-np.clip(Z, -500, 500)))
        return Z

    def backward(self, dA):
        if self.mode == 'relu': return dA * (self.Z > 0)
        if self.mode == 'leaky_relu': 
            dz = np.ones_like(self.Z)
            dz[self.Z <= 0] = 0.01
            return dA * dz
        if self.mode == 'tanh': return dA * (1 - np.tanh(self.Z)**2)
        if self.mode == 'sigmoid':
            s = 1 / (1 + np.exp(-np.clip(self.Z, -500, 500)))
            return dA * (s * (1 - s))
        return dA

class NeuralNetwork:
    def __init__(self, layers_dims, act_type='relu'):
        self.layers = []
        # 根据激活函数自动选择初始化：ReLU族用He，Tanh用Xavier
        init_type = 'he' if 'relu' in act_type else 'xavier'
        for i in range(len(layers_dims) - 1):
            self.layers.append(FullyConnectedLayer(layers_dims[i], layers_dims[i+1], init_type))
            if i < len(layers_dims) - 2:
                self.layers.append(Activation(act_type))

    def forward(self, X):
        for l in self.layers: X = l.forward(X)
        return X

    def backward(self, dZ, l2_lambda):
        for l in reversed(self.layers):
            dZ = l.backward(dZ, l2_lambda) if isinstance(l, FullyConnectedLayer) else l.backward(dZ)

    def update(self, lr, momentum=0.9):
        for l in self.layers:
            if isinstance(l, FullyConnectedLayer):
                l.vW = momentum * l.vW - lr * l.dW
                l.vb = momentum * l.vb - lr * l.db
                l.W += l.vW
                l.b += l.vb