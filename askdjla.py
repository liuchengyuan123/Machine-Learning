import numpy as np
import matplotlib.pyplot as plt

point = np.random.rand(2, 100)
# plt.show()
cl1_x = []
cl1_y = []
cl2_x = []
cl2_y = []
for i in range(100):
    if point[0][i] + point[1][i] < 0.8:
        cl1_x.append(point[0][i])
        cl1_y.append(point[1][i])
    else:
        cl2_x.append(point[0][i])
        cl2_y.append(point[1][i])
plt.figure()
plt.scatter(cl1_x, cl1_y)
plt.scatter(cl2_x, cl2_y)
# plt.show()

id = np.arange(100)
np.random.shuffle(id)

x_in = []
label = []

for i in id:
    x_in.append([point[0][i], point[1][i]])
    if point[0][i] + point[1][i] < 0.8:
        label.append([1])
    else:
        label.append([0])   

x_in = np.array(x_in)
label = np.array(label)

class Identifier:
    def __init__(self, in_dim, out_dim):
        self.w = np.random.rand(in_dim, 1)
        self.b = 0
        self.predict = None
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    def forward(self):
        self.predict = self.sigmoid(np.matmul(x_in, self.w) + self.b)
    def backward(self, lr=0.2):
        grad = -label + self.predict
        grad_w = np.array([np.mean(np.multiply(grad, x_in[:, j])) for j in range(2)])[:, np.newaxis]
        grad_b = np.mean(grad)
        # print(grad_w.shape, self.w.shape)
        self.w -= lr * grad_w
        self.b -= lr * grad_b
    def get_loss(self):
        return -np.mean(np.multiply(label, self.predict) + np.multiply(1 - label, 1 - self.predict))


if __name__ == '__main__':
    iden = Identifier(2, 2)

    for epoch in range(1000):
        iden.forward()
        iden.backward()
        if epoch % 10 == 0:
            print('epoch %d loss %d' % (epoch, iden.get_loss()))
