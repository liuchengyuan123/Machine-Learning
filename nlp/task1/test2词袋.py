import tensorflow as tf
import numpy as np
import os

path = './sentiment labelled sentences'
data = []
label = []
sentense = []

with open(path + '/amazon_cells_labelled.txt', 'r') as f:
    for line in f:
        se, la = line.split('\t')
        # print(line)
        sentense.append(se)
        label.append(la)
        # print('sentence', se, 'label', la)

words = []
for in_ in sentense:
    for x in in_.split():
        for y in x.split(','):
            for z in y.split('.'):
                words.append(z)
words = set(words)
for word in words:
    print(word)
print(len(words))

for se in sentense:
    tmp = []
    for word in words:
        if word in se:
            tmp.append(1)
        else:
            tmp.append(0)
    data.append(tmp)

print('data load in over', len(sentense))

train_data = []
train_label = []
id = np.arange(1000)
np.random.shuffle(id)
test_data = []
test_label = []
for i in id:
    if i < 800:
        if str(label[i]) == 0:
            train_label.append([1, 0])
        else:
            train_label.append([0, 1])
        train_data.append(data[i])
    else:
        if str(label[i]) == 0:
            test_label.append([1, 0])
        else:
            test_label.append([0, 1])
        test_data.append(data[i])
train_label = np.array(train_label)
train_data = np.array(train_data)

print(train_label.shape, train_data.shape)
x_size = train_data.shape

x = tf.placeholder(tf.float32, shape=[None, 2349])
y_ = tf.placeholder(tf.float32, shape=[None, 2])

w = tf.Variable(tf.random_normal([2349, 2]), name='w')
b = tf.Variable(tf.zeros([1, 2]), name='b')
out = tf.add(tf.matmul(x, w), b)
y = tf.sigmoid(out)

cross_entropy = - y_ * tf.log(tf.clip_by_value(y, 1e-10, 1))
loss = tf.reduce_sum(cross_entropy)
optimizer = tf.train.AdamOptimizer(0.005)
train = optimizer.minimize(loss)

predict = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
acc = tf.reduce_mean(tf.cast(predict, "float"))

init = tf.global_variables_initializer()
batch_size = 200
batch_num = 5

with tf.Session() as sess:
    sess.run(init)
    for step in range(500):
        for batch in range(batch_num):
            x_s = train_data[batch * batch_size: (batch + 1) * batch_size]
            y_s = train_label[batch * batch_size: (batch + 1) * batch_size]
            sess.run(train, feed_dict={x: x_s, y_: y_s})
        if step % 50 == 0:
            print('step', step, 'loss', sess.run(loss, feed_dict={x: test_data, y_: test_label}),
                  'accuracy', sess.run(acc, feed_dict={x: test_data, y_: test_label}))
