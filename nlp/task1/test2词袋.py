import tensorflow as tf
import numpy as np

path = './sentiment labelled sentences'
data = []
label = []
sentense = []

with open(path + '/amazon_cells_labelled.txt', 'r') as f:
    for line in f:
        se, la = line.split('\t')
        # print(line)
        sentense.append(se)
        label.append(la)
        # print('sentence', se, 'label', la)

words = []
for in_ in sentense:
    for x in in_.split():
        for y in x.split(','):
            for z in y.split('.'):
                words.append(z)
words = set(words)
'''
for word in words:
    print(word)
print(len(words))
'''
for se in sentense:
    tmp = []
    for word in words:
        if word in se:
            tmp.append(1)
        else:
            tmp.append(0)
    data.append(tmp)

print('data load in over', len(sentense))

train_data = []
train_label = []
id = np.arange(1000)
np.random.shuffle(id)
test_data = []
test_label = []

for j, i in enumerate(id):
    if j < 800:
        if int(label[i]) == 0:
            train_label.append([1, 0])
            train_data.append(data[i])
        else:
            train_label.append([0, 1])
            train_data.append(data[i])
    else:
        if str(label[i]) == 0:
            test_label.append([1, 0])
        else:
            test_label.append([0, 1])
        test_data.append(data[i])
train_label = np.array(train_label)
train_data = np.array(train_data)

print(train_label.shape, train_data.shape)
x_size = train_data.shape

x = tf.placeholder(tf.float32, shape=[None, 2349])
y_ = tf.placeholder(tf.float32, shape=[None, 2])

w = tf.Variable(tf.random_normal([2349, 2]), name='w')
b = tf.Variable(tf.zeros([1, 2]), name='b')
out = tf.add(tf.matmul(x, w), b)
y = tf.sigmoid(out)

cross_entropy = - y_ * tf.log(tf.clip_by_value(y, 1e-10, 1))
loss = tf.reduce_sum(cross_entropy)
optimizer = tf.train.AdamOptimizer(0.15)
train = optimizer.minimize(loss)

predict = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
acc = tf.reduce_mean(tf.cast(predict, tf.float32))

init = tf.global_variables_initializer()
batch_size = 50
batch_num = 20

tf.summary.scalar('accuracy', acc)
merge_summary = tf.summary.merge_all()

with tf.Session() as sess:
    writer = tf.summary.FileWriter('ret/', sess.graph)
    sess.run(init)
    for step in range(500):
        for batch in range(batch_num):
            x_s = train_data[batch * batch_size: (batch + 1) * batch_size]
            y_s = train_label[batch * batch_size: (batch + 1) * batch_size]
            sess.run(train, feed_dict={x: x_s, y_: y_s})

        train_summary = sess.run(merge_summary, feed_dict={x: test_data, y_: test_label})
        writer.add_summary(train_summary, step)
        if step % 50 == 0:
            print('step', step, 'loss', sess.run(loss, feed_dict={x: test_data, y_: test_label}),
                  'accuracy', sess.run(acc, feed_dict={x: test_data, y_: test_label}))
'''
    # input test period
    print('start input test')
    while True:
        line = input()
        answer = int(input())
        print(line)
        print(answer)
        ans = []
        # 1: like
        # 0: not like
        if answer == 0:
            ans.append([1, 0])
        elif answer == 1:
            ans.append([0, 1])
        else:
            break
        ans = np.array(ans)
        tmp = []
        prep = []
        for xx in line.split():
            for yy in xx.split(','):
                for z in yy.split('.'):
                    prep.append(z)
        for word in words:
            if word in prep:
                tmp.append(1)
                print('find', word)
            else:
                tmp.append(0)
        x_in = np.array([tmp])
        print('x_in', x_in, 'ans', ans)
        print(x_in.shape, ans.shape)
        print(sess.run(predict, feed_dict={x: x_in, y_: ans}))
'''
'''
step 0 loss 0.5051048 accuracy 0.575
step 50 loss 0.13044257 accuracy 0.56
step 100 loss 0.092423305 accuracy 0.58
step 150 loss 0.06813327 accuracy 0.595
step 200 loss 0.051323187 accuracy 0.61
step 250 loss 0.039062243 accuracy 0.615
step 300 loss 0.029873652 accuracy 0.625
step 350 loss 0.022864545 accuracy 0.63
step 400 loss 0.017480144 accuracy 0.63
step 450 loss 0.013322119 accuracy 0.635

Process finished with exit code 0

'''
