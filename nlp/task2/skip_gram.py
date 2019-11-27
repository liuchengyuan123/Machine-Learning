import numpy as np
import tensorflow as tf
import os
import re
import random
files = os.listdir('neg')
asd = files[2]
lines = re.split('[.,?\[\]()!~:"]', open('neg/' + asd, 'r').read())
# print(lines[:7])
# print(len(lines))
diction = {}
cur = 0
data_in = []
for line in lines:
	for word in line.split():
		tmp = []
		if word == '':
			continue
		if diction.get(word) is None:
			diction.update({word: cur})
			tmp.append(cur)
			cur += 1
		else:
			tmp.append(diction.get(word))
		data_in.append(tmp)
# print(cur, len(diction))
V = 336
N = 64
C = 5

x = tf.placeholder(dtype=tf.float32, shape=[1, V], name='x')
onehot = tf.placeholder(dtype=tf.float32, shape=[1, V], name='label')

hidden_w = tf.Variable(tf.random.uniform([V, N]), tf.float32, name='hidden_w')
hidden_out = tf.reduce_mean(tf.matmul(x, hidden_w), 0)

out_w = tf.Variable(tf.random.uniform([N, V]), tf.float32)
y_ = tf.nn.softmax(tf.matmul(tf.reshape(hidden_out, [1, N]), out_w))

alpha = 0.2
entropy = - tf.reduce_mean(onehot * tf.log(tf.clip_by_value(y_, 1e-10, 1)))
optimizer = tf.train.GradientDescentOptimizer(alpha)
train = optimizer.minimize(entropy)

init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	print('init value:')
	print(sess.run(hidden_w, feed_dict={}))
	for step in range(100):	
		for line in data_in:
			for c in range(len(line)):
				dat = [[0.] * V]
				lab = [[0.] * V]
				for t in range(c, min(len(line), c + C)):
					if t is c + C // 2:
						dat[0][line[t]] = 1.
					else:
						lab[0][line[t]] = 1.
				dat = np.array(dat)
				lab = np.array(lab)

				sess.run(train, feed_dict={x: dat, onehot: lab})
				print('step %d, loss %f' % (step, sess.run(entropy, feed_dict={x: dat, onehot: lab})))
		if step % 10 is 0:
			print('----------step %d' % step)
			print(sess.run(hidden_w, feed_dict={}))
