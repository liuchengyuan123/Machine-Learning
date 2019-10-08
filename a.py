import numpy as np
import matplotlib.pyplot as plt

data_in = 10 * np.random.rand(500, 2)

p1x = []
p1y = []
p2x = []
p2y = []
label = []
for d in data_in:
    if d[0] + d[1] < 10:
        p1x.append(d[0])
        p1y.append(d[1])
        label.append([0])
    else:
        p2x.append(d[0])
        p2y.append(d[1])
        label.append([1])
plt.figure()
plt.scatter(p1x, p1y)
plt.scatter(p2x, p2y)
plt.show()

data_in = np.array(data_in)
label = np.array(label)

def sigmoid(x_in):
    return 1 / (1 + np.exp(-x_in))

def forward(w, b, data):
    return sigmoid(np.matmul(data, w) + b)

def get_grad(x, predict, label):
    error = - (predict - label)
    grad_w = []
    for i in range(2):
        # print(error.shape, x[:, i][:, np.newaxis].shape)
        grad_w.append([np.mean(error * x[:, i][:, np.newaxis])])
    grad_b = np.mean(error)
    return np.array(grad_w), grad_b

def get_loss(predict, y):
    return - np.mean(y * np.log(np.clip(predict, 1e-20, 1)) + (1 - y) * np.log(np.clip(1 - predict, 1e-20, 1)))

w = np.random.rand(2, 1)
b = 0
alpha = 0.1
for epoch in range(1000):
    predict = forward(w, b, data_in)
    grad_w, grad_b = get_grad(data_in, predict, label)
    w += alpha * grad_w
    b += alpha * grad_b
    if epoch % 100 == 0:
        print('loss', get_loss(predict, label))
print(w, b)
# print(predict)
p1x = []
p1y = []
p2x = []
p2y = []
label = []
for i, d in enumerate(data_in):
    if predict[i] < 0.5:
        p1x.append(d[0])
        p1y.append(d[1])
    else:
        p2x.append(d[0])
        p2y.append(d[1])
plt.figure()
plt.scatter(p1x, p1y)
plt.scatter(p2x, p2y)
plt.show()

