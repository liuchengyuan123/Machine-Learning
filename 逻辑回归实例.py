import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('data/MNIST_data', one_hot=True)

batch_size = 100    # train batch size
batch_num = mnist.train.num_examples // batch_size  # number of batches

# two placeholder
x = tf.placeholder(tf.float32, [None, 784], name='x')
y = tf.placeholder(tf.float32, [None, 10], name='y')

w = tf.Variable(tf.zeros([784, 10]), name='weights')
b = tf.Variable(tf.zeros([10]), name='bias')

predict = tf.nn.sigmoid(tf.matmul(x, w, name='weight_x') + b)

loss = tf.reduce_mean(tf.square(y - predict))
optimizer = tf.train.GradientDescentOptimizer(0.3)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()    # initialize all variables
correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(predict, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tf.summary.histogram('accuracy', accuracy)
tf.summary.scalar('accuracy', accuracy)
merge_summary = tf.summary.merge_all()

with tf.Session() as sess:
    writer = tf.summary.FileWriter('train_ret/', sess.graph)
    sess.run(init)
    for epoch in range(21):
        for batch in range(batch_size):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            sess.run(train, feed_dict={x: batch_x, y: batch_y})

        train_summary = sess.run(merge_summary, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        writer.add_summary(train_summary, epoch)
        acc = sess.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels})
        print('Iter', str(epoch), 'test accuracy', str(acc))
