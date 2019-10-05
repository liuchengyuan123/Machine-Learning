import tensorflow as tf
import numpy as np
import os

path = './aclImdb/train'
files = os.listdir(path + '/pos')

train_pos_words = []

for id, file in enumerate(files):
    if id >= 1000:
        break
    fp = open(path + '/pos/' + file, 'r')
    train_pos_words.append(fp.read())

print('pos data load over', train_pos_words.__len__())

files = os.listdir(path + '/neg')
train_neg_words = []

for id, file in enumerate(files):
    if id >= 1000:
        break
    fp = open(path + '/neg/' + file, 'r')
    train_neg_words.append(fp.read())
print('neg data load over', train_neg_words.__len__())

train_pos_array = []
train_neg_array = []

mymap = {}

for sentense in train_pos_words:
    tmp = []
    for words_in in sentense.split():
        for ww in words_in.split('.'):
            for w in ww.split(','):
                for www in w.split('"'):
                    for ws in www.split('?'):
                        for tmd in ws.split('!'):
                            for on in tmd.split('('):
                                for off in on.split(')'):
                                    tmp.append(off)
                                    if off not in mymap:
                                        mymap[off] = 1
                                    else:
                                        mymap[off] += 1
    train_pos_array.append(tmp)

for sentense in train_neg_words:
    tmp = []
    for words_in in sentense.split():
        for ww in words_in.split('.'):
            for w in ww.split(','):
                for www in w.split('"'):
                    for ws in www.split('?'):
                        for tmd in ws.split('!'):
                            for on in tmd.split('('):
                                for off in on.split(')'):
                                    tmp.append(off)
                                    if off not in mymap:
                                        mymap[off] = 1
                                    else:
                                        mymap[off] += 1
    train_neg_array.append(tmp)

mymap[''] = 0
# print('unique over, with len', words.__len__())

waiters = sorted(mymap.items(), key = lambda x:x[1], reverse = True)
waiters = waiters[:1500]
tba = []
for word in waiters:
    # print(word[0])
    tba.append(word[0])

train_pos_words = []
train_neg_words = []
llen = len(tba)
for sentense in train_pos_array:
    tmp = []
    for word in tba:
        if word in sentense:
            tmp.append(1)
        else:
            tmp.append(0)
    train_pos_words.append(tmp)

for sentense in train_neg_array:
    tmp = []
    for word in tba:
        if word in sentense:
            tmp.append(1)
        else:
            tmp.append(0)
    train_neg_words.append(tmp)

id = np.arange(2000)
np.random.shuffle(id)
train_data = []
train_label = []
for i in id:
    if i >= 1000:
        train_label.append([0, 1])
        train_data.append(train_neg_words[i - 1000])
    else:
        train_label.append([1, 0])
        train_data.append(train_pos_words[i])
train_data = np.array(train_data)
train_label = np.array(train_label)
print(train_data.shape, train_label.shape)
print('train data and labels prepared over')

x = tf.placeholder(tf.float32, shape=[None, 1500])
y = tf.placeholder(tf.float32, shape=[None, 2])

w = tf.Variable(tf.random_normal([1500, 2]), tf.float32)
b = tf.Variable(tf.zeros([1, 2]), tf.float32)
out = tf.add(tf.matmul(x, w), b)
y_ = tf.sigmoid(out)

cross_entropy = - y * tf.log(tf.clip_by_value(y_, 1e-10, 1))
loss = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(0.1)
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
            sess.run(train, feed_dict={x: x_s, y: y_s})
        if step % 50 == 0:
            print('step', step, 'loss', sess.run(loss, feed_dict={x: x_s, y: y_s}), 'accuracy', sess.run(acc, feed_dict={x: x_s, y: y_s}))
