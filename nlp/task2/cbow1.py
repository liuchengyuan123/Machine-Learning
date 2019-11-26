import numpy as np
import tensorflow as tf
import os
import re

pos_files = os.listdir('pos')
neg_files = os.listdir('neg')

diction = {}
size = 0
sentenses = [] # sentenses dataset
for com in pos_files:
	name = 'pos/' + com
	try:
		with open(name, 'r') as f:
			text = f.read()
			for sen in re.split('[.,?"()!:\]\[]', text):
				if sen == '':
					continue
				sentenses.append(sen)
				for word in sen.split():
					if word == '':
						continue
					if diction.get(word) is None:
						diction[word] = size
						size += 1
	except:
		continue
for com in pos_files:
	name = 'neg/' + com
	try:
		with open(name, 'r') as f:
			text = f.read()
			for sen in re.split('[.,?"()!:\]\[]', text):
				if sen == '':
					continue
				sentenses.append(sen)
				for word in sen.split():
					if word == '':
						continue
					if diction.get(word) is None:
						diction[word] = size
						size += 1
	except:
		continue
print(len(diction))
V = len(diction)
C = 4
N = 64
'''
generate one-hot code
'''

x = tf.placeholder(dtype=tf.float32, shape=[C, V], name='x')
label = tf.placeholder(dtype=tf.float32, shape=[1, V], name='label')

hid_w  = tf.Variable(tf.random.uniform([V, N]), dtype=tf.float32, name='hidden_layer_weight')
hid_out = tf.reduce_mean(tf.matmul(x, hid_w, name='hidden_layer_output'), 0)

out_w = tf.Variable(tf.random.uniform([N, V]) ,dtype=tf.float32, name='output_weight')
y_ = tf.nn.softmax(tf.matmul(tf.reshape(hid_out, [1, N]), out_w, name='output_layer'))

alpha = 0.2
entropy = - tf.reduce_mean(label * tf.log(tf.clip_by_value(y_, 1e-10, 1)))
optimizer = tf.train.GradientDescentOptimizer(alpha)
train = optimizer.minimize(entropy)

init = tf.global_variables_initializer()

print(len(sentenses))
with tf.Session() as sess:
	sess.run(init)

	for step in range(1):
		for idd in range(len(sentenses)):
			line = sentenses[idd]
			words = line.split()
			# words.remove('')
			for c in range(len(words)):
				data = []
				lab = [[0] * V]
				for p in range(C):
					data.append([0] * V)
				for t in range(c, min(len(words), c + C)):
					word = diction[words[t]]
					data[t - c][word] = 1.
					lab[0][word] = 1.
					# print(word, words[t])
				data = np.array(data)
				lab = np.array(lab)
				
				sess.run(train, feed_dict={x: data, label: lab})
				
				if step % 1 == 0 and line == sentenses[-1] and c == len(words) - 1:
					print('step %d----------' % step)
					print('loss %f' % (sess.run(entropy, feed_dict={x: data, label: lab})))
			if idd % 10 == 0:
				print('idd %d' % idd)
