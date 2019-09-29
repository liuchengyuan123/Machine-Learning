import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x_input = np.random.rand(500, 2)

point_x = [[], [], []]
point_y = [[], [], []]

labels = []

for i in range(500):
    if x_input[i][0] + x_input[i][1] < 0.7:
        point_x[0].append(x_input[i][0])
        point_y[0].append(x_input[i][1])
        labels.append([1, 0, 0])
    elif x_input[i][0] + x_input[i][1] > 1.2:
        point_x[2].append(x_input[i][0])
        point_y[2].append(x_input[i][1])
        labels.append([0, 1, 0])
    else:
        point_x[1].append(x_input[i][0])
        point_y[1].append(x_input[i][1])
        labels.append([0, 0, 1])

plt.figure()
plt.scatter(point_x[0], point_y[0])
plt.scatter(point_x[1], point_y[1])
plt.scatter(point_x[2], point_y[2])
# plt.show()

test_input = x_input[:300]
test_labels = labels[:300]

train_input = x_input[300:]
train_labels = labels[300:]

x = tf.placeholder(tf.float32, shape=[None, 2], name='x')
y_ = tf.placeholder(tf.float32, shape=[None, 3], name='y')

weight_layer1 = tf.Variable(tf.zeros([2, 10]), dtype=tf.float32, name='weight_layer1')
bias_layer1 = tf.Variable(tf.zeros([1, 10]), dtype=tf.float32, name='bias_layer1')
output_layer1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weight_layer1, name='output_layer1'), bias_layer1))

weight_layer2 = tf.Variable(tf.zeros([10, 3]), dtype=tf.float32, name='weight_layer2')
bias_layer2 = tf.Variable(tf.zeros([1, 3]), dtype=tf.float32, name='bias_layer2')
y = tf.nn.sigmoid(tf.add(tf.matmul(output_layer1, weight_layer2, name='output_layer2'), bias_layer2))

cross_entropy = - y_ * tf.log(tf.clip_by_value(y, 1e-10, 1)) - (1 - y_) * tf.log(tf.clip_by_value(1 - y, 1e-10, 1))
loss = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(0.2)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

arv = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
acc = tf.reduce_mean(tf.cast(arv, 'float'))

tf.summary.scalar('loss', loss)
tf.summary.scalar('accuacy', acc)

merge_summary = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(init)

    writer = tf.summary.FileWriter('test/', sess.graph)

    for step in range(200):
        sess.run(train, feed_dict={x: train_input, y_: train_labels})
        train_summary = sess.run(merge_summary, feed_dict={x: train_input, y_: train_labels})
        writer.add_summary(train_summary, step)

        if step % 10 == 0:
            print('step number %d with -------' % step, end=' test loss: ')
            print(sess.run(loss, feed_dict={x: train_input, y_: train_labels}), end=', train accuracy: ')
            print(sess.run(acc, feed_dict={x: test_input, y_: test_labels}))
    
