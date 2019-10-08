import numpy as np
import matplotlib.pyplot as plt

class Dense:

    def __init__(self, in_dim, out_dim):
        m, self.n = in_dim
        om, self.on = out_dim
        self.w = np.random.rand(self.n, self.on)
        self.b = np.random.rand(1, self.on)
        self.predict = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def forward(self, data_in):
        self.predict = self.sigmoid(np.matmul(data_in, self.w) + self.b)
        return self.predict

    def get_grad(self, data_in, predict, label):
        error = - (predict - label)
        grad_w = []
        for i in range(self.n):
            grad_w.append([np.mean(error * data_in[:, i][:, np.newaxis])])
        grad_b = np.mean(error)
        return np.array(grad_w), grad_b
    
    def backward(self, data_in, label, alpha):
        grad_w, grad_b = self.get_grad(data_in, self.predict, label)
        self.w += alpha * grad_w
        self.b += alpha * grad_b

    def train(self, data_in, label, alpha=0.2):
        self.forward(data_in)
        self.backward(data_in, label, alpha)
    
    def get_loss(self, predict, label):
        return -np.mean(label * np.log(np.clip(predict, 1e-10, 1)) + (1 - label) * np.log(np.clip(1 - predict, 1e-10, 1)))

data_in = 10 * np.random.rand(500, 2)
label = []
for d in data_in:
    if d[0] + 2 * d[1] < 10:
        label.append([0])
    else:
        label.append([1])
label = np.array(label)
alpha = 0.2
lcy = Dense((None, 2), (None, 1))
for epoch in range(1000):
    lcy.train(data_in, label, alpha)
    if epoch % 10 == 0:
        print('epoch', epoch, 'loss', lcy.get_loss(lcy.predict, label))

p1x = []
p1y = []
p2x = []
p2y = []
for i, d in enumerate(data_in):
    if lcy.predict[i] < 0.5:
        p1x.append(d[0])
        p1y.append(d[1])
    else:
        p2x.append(d[0])
        p2y.append(d[1])
plt.figure()
plt.scatter(p1x, p1y)
plt.scatter(p2x, p2y)
plt.show()
