import tensorflow as tf
import numpy as np
import re

'''
使用N特征
用正则表达式提取单词
在每个句子的一个特征列表中，提取连续的N个单词作为向量
转换大小写

猜想：或许可以字典树优化
（Huffman tree也可以）

这种方法表示词向量非常复杂，而且运算量巨大，没有足够多的词向量参与矩阵（否则运算很慢），导致没有出现过的词向量的情绪根本无法判断
我认为这是限制正确率的一个主要因素
'''

path = './sentiment labelled sentences'
label = []
sentense = []

with open(path + '/yelp_labelled.txt', 'r') as f:
    for line in f:
        se, la = line.split('\t')
        # print(line)
        se = se.lower()
        sentense.append(se)
        label.append(int(la))
        # print('sentence', se, 'label', la)

words = []
sen = [] # 每一个句子的单词顺序，以单词形式存储
for x in sentense:
    tmp = []
    for y in x.split():
        for w in re.split('[ ,".?!:()]', y):
            tmp.append(w)
    for u in tmp:
        words.append(u)
    sen.append(tmp)

words = list(set(words)) # 单词集合，虽然没什么用，但是先放在这
for word in words:
    print(word)
print(len(words))
sen_num = [] # 每一个句子的单词顺序，以数字形式存储
for s in sen:
    tmp = []
    for t in s:
        tmp.append(words.index(t)) # 不知道是不是哈希之后二分找的，应该log1000级别
    sen_num.append(tmp)
N = 3
fea_vector = [] # 每一句话的N特征向量
tot_fea = [] # 所有的N特征向量
for j, Dat in enumerate(sen_num):
    tmp = []
    # if len(Dat) <= N:
    #     print(j, 'empty', sen[j])
    for i in range(N - 1):
        Dat.append(1912)
    for x in range(len(Dat) - N + 1):
        # fea_vector.append(Dat[x: x + N])
        # 由于列表没有hash方法，不能直接放进去查看有多少总特征，所以自定义一个hash方法
        # 总共1912个单词，把所有单词哈希成四位数，字符串拼接起来
        nows = ''
        for i in range(N):
            nows += ('%04d' % (Dat[x + i]))
        assert(len(nows) == 4 * N)
        tot_fea.append(nows)
        tmp.append(nows)
    fea_vector.append(tmp)
tot_fea = list(set(tot_fea))
# for j in tot_fea:
#     print(j)
print(len(tot_fea))
# 总共产生了6390个特征向量

data = []
for a in fea_vector:
    tmp = []
    for b in tot_fea:
        tmp.append(1 if b in a else 0)
    data.append(tmp)
    if 1 not in tmp:
        print('problem', a)
print('data load in over')
train_data = []
train_label = []
test_data = []
test_label = []

for id, d in enumerate(data):
    if id < 950:
        train_data.append(d)
        if label[id] == 1:
            train_label.append([0, 1])
        else:
            train_label.append([1, 0])
    else:
        test_data.append(d)
        if label[id] == 1:
            test_label.append([0, 1])
        else:
            test_label.append([1, 0])
train_data = np.array(train_data)
train_label = np.array(train_label)
test_data = np.array(test_data)
test_label = np.array(test_label)

x = tf.placeholder(tf.float32, shape=[None, 10045])
y_ = tf.placeholder(tf.float32, shape=[None, 2])

w = tf.Variable(tf.random_normal([10045, 2]), name='w')
b = tf.Variable(tf.zeros([1, 2]), name='b')
out = tf.add(tf.matmul(x, w), b)
y = tf.sigmoid(out)
#
# w_2 = tf.Variable(tf.random_normal([10, 2]), tf.float32)
# b_2 = tf.Variable(tf.zeros([1, 2]), tf.float32)
# out2 = tf.add(tf.matmul(layer1, w_2), b_2)
# y = tf.sigmoid(out2)

cross_entropy = - y_ * tf.log(tf.clip_by_value(y, 1e-10, 1))
loss = tf.reduce_sum(cross_entropy)
optimizer = tf.train.AdamOptimizer(0.02)
train = optimizer.minimize(loss)

predict = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
acc = tf.reduce_mean(tf.cast(predict, tf.float32))

init = tf.global_variables_initializer()
batch_size = 50
batch_num = 20

# tf.summary.scalar('accuracy', acc)
# merge_summary = tf.summary.merge_all()

with tf.Session() as sess:
    writer = tf.summary.FileWriter('ret/', sess.graph)
    sess.run(init)
    for step in range(1000):
        for batch in range(batch_num):
            x_s = train_data[batch * batch_size: (batch + 1) * batch_size]
            y_s = train_label[batch * batch_size: (batch + 1) * batch_size]
            sess.run(train, feed_dict={x: x_s, y_: y_s})

        # train_summary = sess.run(merge_summary, feed_dict={x: test_data, y_: test_label})
        # writer.add_summary(train_summary, step)
        if step % 50 == 0:
            print('step', step, 'train_accuracy', sess.run(acc, feed_dict={x: train_data, y_: train_label}),
                  'test accuracy', sess.run(acc, feed_dict={x: test_data, y_: test_label}))
'''
step 0 train_accuracy 0.53894734 test accuracy 0.66
step 50 train_accuracy 0.7957895 test accuracy 0.62
step 100 train_accuracy 0.8326316 test accuracy 0.62
step 150 train_accuracy 0.85263157 test accuracy 0.62
step 200 train_accuracy 0.8673684 test accuracy 0.62
step 250 train_accuracy 0.8810526 test accuracy 0.62
step 300 train_accuracy 0.8905263 test accuracy 0.64
step 350 train_accuracy 0.9031579 test accuracy 0.66
step 400 train_accuracy 0.9157895 test accuracy 0.66
step 450 train_accuracy 0.9178947 test accuracy 0.66
step 500 train_accuracy 0.92421055 test accuracy 0.66
step 550 train_accuracy 0.93157893 test accuracy 0.66
step 600 train_accuracy 0.93578947 test accuracy 0.66
step 650 train_accuracy 0.94631577 test accuracy 0.66
step 700 train_accuracy 0.951579 test accuracy 0.66
step 750 train_accuracy 0.9526316 test accuracy 0.66
step 800 train_accuracy 0.9568421 test accuracy 0.66
step 850 train_accuracy 0.96105266 test accuracy 0.66
step 900 train_accuracy 0.9631579 test accuracy 0.66
step 950 train_accuracy 0.9642105 test accuracy 0.66
'''

