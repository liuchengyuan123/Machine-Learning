import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

data_set = np.random.rand(800, 2)
print(data_set.shape)

x_input = []
y_input = []
for i in range(800):
    x_input.append(data_set[i][0])
    y_input.append(data_set[i][1])

x_input = np.array(x_input)
y_input = np.array(y_input)

ok_x = []
ok_y = []
labels = []
for i in range(800):
    if x_input[i] + y_input[i] < 0.8:
        ok_x.append(x_input[i])
        ok_y.append(y_input[i])
        labels.append(1)
    else:
        labels.append(0)
labels = np.array(labels)[:, np.newaxis]

ok_x = np.array(ok_x)
ok_y = np.array(ok_y)
print('ok_X.shape', ok_x.shape, 'ok_y.shape', ok_y.shape)
x_axis = np.linspace(0, 1, 500)[:, np.newaxis]
line = 0.8 - x_axis
plt.figure()
plt.scatter(x_input, y_input)
plt.scatter(ok_x, ok_y, c='r')
plt.plot(x_axis, line, 'g-', lw=3)
plt.show()

x = tf.placeholder(tf.float32, shape=[None, 2])
y_ = tf.placeholder(tf.float32, shape=[None, 1])

weight_layer1 = tf.Variable(tf.zeros([2, 10]), name='weight_layer1')
bias_layer1 = tf.Variable(tf.zeros([1, 10]), name='bias_layer1')
ret_1 = tf.add(tf.matmul(x, weight_layer1, name='out_layer1'), bias_layer1)
out_layer1 = tf.sigmoid(ret_1)

weight_layer2 = tf.Variable(tf.zeros([10, 1]), dtype=tf.float32, name='weight_layer2')
bias_layer2 = tf.Variable(tf.zeros([1, 1]), dtype=tf.float32, name='bias_layer2')
ret_2 = tf.add(tf.matmul(out_layer1, weight_layer2, name='out_layer2'), bias_layer2)
y = tf.sigmoid(ret_2)

cross_entropy = - y_ * tf.log(tf.clip_by_value(y, 1e-10, 1)) - (1 - y_) * tf.log(tf.clip_by_value(1 - y, 1e-10, 1))
loss = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(0.2)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

# tf.summary.scalar('loss', loss)
# merge_summary = tf.summary.merge_all()

train_data = np.random.rand(30, 2)
train_labels = np.array([int(x1 +x2 <= 0.8) for (x1, x2) in train_data])[:, np.newaxis]
print(train_labels.shape)

a_point_x = []
a_point_y = []
b_point_x = []
b_point_y = []

for i in range(30):
    if train_labels[i] == 1:
        a_point_x.append(train_data[i][0])
        a_point_y.append(train_data[i][1])
    else:
        b_point_x.append(train_data[i][0])
        b_point_y.append(train_data[i][1])
plt.scatter(a_point_x, a_point_y, c='r')
plt.scatter(b_point_x, b_point_y, c='g')
plt.show()

with tf.Session() as sess:
    sess.run(init)

#    writer = tf.summary.FileWriter('test/', sess.graph)

    for step in range(1000):
        sess.run(train, feed_dict={x: data_set, y_: labels})

#       train_summary = sess.run(merge_summary, feed_dict={x: data_set, y_: labels})
#       writer.add_summary(train_summary, step)
        if step % 100 == 0:
            print('step:', step, '-------------')
            print(sess.run(loss, feed_dict={x: data_set, y_: labels}))
    print('train result:------------')
    print(sess.run(loss, feed_dict={x: train_data, y_: train_labels}))
