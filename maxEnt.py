import tensorflow as tf
import numpy as np


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def read_file(file):
    feats = []
    wrds = []

    with open(file) as inpt:
        for line in inpt:
            split = line.strip().split()
            feat_vec = [float(val) for val in split[:-1]]
            feats.append(feat_vec)
            wrd = split[-1]
            wrds.append(wrd)

    return feats, wrds


def read_vocab(l2):
    vcb = {}
    with open('vocab_test.' + l2) as voc:
        for i, line in enumerate(voc):
            vcb[line.strip().split()[0]] = i
    return vcb


lang1, lang2 = 'bul', 'rus'
vocab = read_vocab(lang2)
num_classes = len(vocab)
features, words = read_file('vectors_test.' + lang1 + '.' + lang2)
one_hots = []
for w in words:
    one_hot = [0] * num_classes
    one_hot[vocab[w]] = 1
    one_hots.append(one_hot)
num_features = len(features[0])
num_tokens = len(features)

# graph
x = tf.placeholder(tf.float32, [None, num_features], name='x')
y_true = tf.placeholder(tf.float32, [None, num_classes], name='y_true')

# model
W = tf.Variable(tf.random_normal([num_features, num_classes], stddev=0.01), name='W')
b = tf.Variable(tf.zeros([num_classes]), name='b')
y = tf.nn.softmax(tf.add(tf.matmul(x, W), b))


def log2(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(2, dtype=numerator.dtype))
    return numerator / denominator


# training
cross_entropy = -tf.reduce_sum(y_true * log2(y))
cost_function = tf.reduce_mean(cross_entropy)
learning_rate = 0.001
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost_function)

# launch session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    num_epochs = 100
    batch_size = 100

    for j in range(num_epochs):
        np.random.shuffle(features)
        cost = 0.0
        for batch_x, batch_y in zip(chunks(features, batch_size), chunks(one_hots, batch_size)):
            _, c = sess.run([optimizer, cost_function], feed_dict={x: np.array(batch_x), y_true: np.array(batch_y)})
            cost += c

        print("after epoch:", j)
        print("W:", sess.run(tf.reduce_mean(W, 1)))
        # print("b:", sess.run(b))
        print('cost:', cost)
        perplexity = np.power(cost, 2)
        print("perplexity:", perplexity)
