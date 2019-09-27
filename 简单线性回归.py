import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x_data = np.linspace(-0.5, 0.5, 1000)[:, np.newaxis]
noise = np.random.normal(0, 0.02, x_data.shape)
y_data = x_data * 3.3 + noise + 1.

x = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='x')
y = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='y')

k = tf.Variable(tf.random_normal([1]), name='k')
b = tf.Variable(0., name='b')
prediction = k * x + b

init = tf.global_variables_initializer()
loss = tf.reduce_mean(tf.square(y - prediction))
optimizer = tf.train.GradientDescentOptimizer(0.2)
train = optimizer.minimize(loss)

tf.summary.scalar('loss', loss)
tf.summary.histogram('loss', loss)
merge_summary = tf.summary.merge_all()
with tf.Session() as sess:
    writer = tf.summary.FileWriter('train_ret/', sess.graph)
    sess.run(init)

    for step in range(1000):
        sess.run(train, feed_dict={x: x_data, y: y_data})
        train_summary = sess.run(merge_summary, feed_dict={x: x_data, y: y_data})
        writer.add_summary(train_summary, step)

        if step % 100 == 0:
            print('step:', step, '----------')
            print(sess.run(loss, feed_dict={x: x_data, y: y_data}))

    kval = sess.run(k)
    bval = sess.run(b)
    print('k', k.eval(), 'b', b.eval())

    plt.figure()
    plt.scatter(x_data, y_data)
    prediction_val = sess.run(prediction, feed_dict={x: x_data})
    plt.plot(x_data, prediction_val, 'r-', lw=3)
    plt.show()
