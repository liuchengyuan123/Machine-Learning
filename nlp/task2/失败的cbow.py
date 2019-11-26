import tensorflow as tf
import random
import numpy as np
import os
import re

'''
上下文环境预测中心词
'''

files = os.listdir('pos')
lines = []
diction = {}
cur = 0
for file in files:
	name = 'pos/' + file
	with open(name, 'r') as f:
		sen = f.read()
		for line in re.split('[,.?!":(\[\])]', sen):
			lines.append(line)
			for word in line.split():
				if diction.get(word) == None:
					diction.update({word: cur})
					cur += 1
	break
# print(len(diction), cur)
V = 660
C = 5
N = 32

x = tf.placeholder(dtype=tf.float32, shape=[C, V], name='x')
label = tf.placeholder(dtype=tf.float32, shape=[1, V], name='label')

hidden_w = tf.Variable(tf.random.uniform([V, N]), name='hidden_layer_weight')
hidden_out = tf.reduce_sum(tf.matmul(x, hidden_w, name='hidden_layer_output'), 0)

out_w = tf.Variable(tf.random.uniform([N, V]), name='output_weight')
y_ = tf.nn.softmax(tf.matmul(tf.reshape(hidden_out, [1, N]), out_w, name='output_layer'))

alpha = 0.2
entropy = - tf.reduce_sum(label * tf.log(tf.clip_by_value(y_, 1e-10, 1)))
optimizer = tf.train.AdamOptimizer(alpha)
train = optimizer.minimize(entropy)

init = tf.global_variables_initializer()
print(diction)
f = False
with tf.Session() as sess:
	sess.run(init)

	for step in range(100):
		for line in lines:
			words = line.split()
			u = random.randint(0, len(words))
			for c in range(len(words)):
				data = []
				for t in range(C):
					data.append([0] * V)
				lab = [[0] * V]
				for t in range(c, min(c + C, len(words))):
					if t == c + C // 2:
						lab[0][diction[words[t]]] = 1.
						continue
					data[t - c][diction[words[t]]] = 1.
				data = np.array(data)
				lab = np.array(lab)
				
				sess.run(train, feed_dict={x: data, label: lab})
				if f is True and c == u:
					print('step %d, loss %f' % (step, sess.run(entropy, feed_dict={x: data, label: lab})))
					print(sess.run(y_, feed_dict={x: data}))
					print(lab, lab.shape)
					print('debug %d' % diction[words[c]], words[c], line)
					f = False

		if step % 10 == 0:
			random.shuffle(lines)
			print('debug hidden_w')
			print(sess.run(hidden_w, feed_dict={}))
			f = True
