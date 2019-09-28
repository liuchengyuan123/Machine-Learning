import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x_input = np.random.rand(700, 1)
y_input = np.random.rand(700, 1)

print(x_input, y_input)

x_axis = np.linspace(0, 1, 500)
y_data = 0.7 - x_axis

print('x:', x_axis, 'y:', y_data)

plt.figure()
plt.scatter(x_input, y_input)	
plt.plot(x_axis, y_data, 'r-')
# plt.show()

x_train = []

for i in range(x_input.__len__()):
	x_train.append([x_input[i][0], y_input[i][0]])
x_train = np.array(x_train)
# print('x_train:', x_train, 'shape:', x_train.shape)

labels = np.array([int(x1 + x2 < 0.7) for (x1, x2) in x_train])[:, np.newaxis]
print('labels:', labels.shape)

pointx = []
pointy = []

for i in range(700):
	if labels[i][0] == 1:
		pointx.append(x_input[i][0])
		pointy.append(y_input[i][0])
plt.scatter(pointx, pointy)
plt.show()	
x = tf.placeholder(tf.float32, shape=[None, 2], name='x-input')
y_ = tf.placeholder(tf.float32, shape=[None, 1], name='y-input')

w = tf.Variable(tf.zeros([2, 1]), tf.float32, name='weight')
b = tf.Variable(tf.zeros([1, 1]), tf.float32, name='bias')

y = tf.add(tf.matmul(x, w, name='add'), b, name='y')
y_hat = tf.sigmoid(y)

cross_entropy = - (y_ * tf.log(tf.clip_by_value(y_hat, 1e-10, 1)) + (1 - y_) * tf.log(tf.clip_by_value(1 - y_hat, 1e-10, 1)))
loss = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(0.2)
train_step = optimizer.minimize(loss)

init = tf.global_variables_initializer()

tf.summary.scalar('loss', loss)
merge_summary = tf.summary.merge_all()

with tf.Session() as sess:
	sess.run(init)

	writer = tf.summary.FileWriter('test/', sess.graph)

	for step in range(900):
		sess.run(train_step, feed_dict={x: x_train, y_: labels})
		
		train_summary = sess.run(merge_summary, feed_dict={x: x_train, y_: labels})
		writer.add_summary(train_summary, step)

		if step % 15 == 0:
			print('step:', step, '----------')
			print('w', sess.run(w), 'b', sess.run(b), 'loss', sess.run(loss, feed_dict={x: x_train, y_: labels}))
	wval = sess.run(w)
	bval = sess.run(b)
	

