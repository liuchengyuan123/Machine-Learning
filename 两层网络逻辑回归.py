import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('data/', one_hot=True)

train_img = mnist.train.images
train_label = mnist.train.labels
test_img = mnist.test.images
test_label = mnist.test.labels

x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
y = tf.placeholder(tf.float32, shape=[None, 10], name='y')

w1 = tf.Variable(tf.zeros([784, 30]), dtype=tf.float32, name='w1')
b1 = tf.Variable(tf.zeros([1, 30]), dtype=tf.float32, name='b1')
out1 = tf.nn.sigmoid(tf.add(tf.matmul(x, w1, name='mul_layer1'), b1, name='out1'))

w2 = tf.Variable(tf.zeros([30, 10]), name='w2')
b2 = tf.Variable(tf.zeros([1, 10]), name='b2')
y_ = tf.nn.sigmoid(tf.add(tf.matmul(out1, w2, name='mul_layer2'), b2, name='y'))

corss_entropy = - y * tf.log(tf.clip_by_value(y_, 1e-10, 1)) - (1 - y) * tf.log(tf.clip_by_value(1 - y_, 1e-10, 1))
loss = tf.reduce_mean(corss_entropy)
optimizer = tf.train.AdamOptimizer(0.001)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

pred = tf.equal(tf.argmax(y, 1),tf.argmax(y_, 1))
acc = tf.reduce_mean(tf.cast(pred,"float"))

training_epochs = 50  # 样本迭代次数
batch_size = 100 # 每次迭代使用的样本
display_step = 10

tf.summary.scalar('train accuracy', acc)
tf.summary.scalar('train loss', loss)
merge_summary = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter('test/', sess.graph)

    for epoch in range(training_epochs):
        avg_cost = 0.
        number_batch = int(mnist.train.num_examples/batch_size)
        for i in range(number_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train,feed_dict={x:batch_xs,y:batch_ys})
            feeds = {x: batch_xs,y: batch_ys}
            avg_cost += sess.run(loss,feed_dict=feeds)/number_batch
        # DISPLAY

        train_summary = sess.run(merge_summary, feed_dict={x: batch_xs, y: batch_ys})
        writer.add_summary(train_summary, epoch)

        if epoch % display_step == 0:
            feeds_train = {x:batch_xs,y:batch_ys}
            feeds_test = {x: mnist.test.images,y:mnist.test.labels}
            train_acc = sess.run(acc,feed_dict=feeds_train)
            test_acc = sess.run(acc,feed_dict=feeds_test)
            print(("Epoch: %03d/%03d cost:%.9f train_acc: %.3f test_acc: %.3f") % (epoch,training_epochs,avg_cost,train_acc,test_acc))
    print("Done")
