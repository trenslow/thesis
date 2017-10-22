import tensorflow as tf
import numpy as np
import itertools
import time
import random
import sys


def chunks(l, n):
    # for efficient iteration over whole data set
    for i in range(0, len(l), n):
        yield l[i:i + n]


def read_file(file):
    feats = []
    wrds = []
    vcb = {}
    
    with open(file) as inpt:
        for line in inpt:  # for looping over whole dataset
        # for line in itertools.islice(inpt, 677500):  # for taking first n lines of dataset (for development)
            split = line.strip().split()
            feat_vec = [float(val) for val in split[:-1]]
            feats.append(feat_vec)
            wrd = split[-1]
            wrds.append(wrd)
            if wrd not in vcb:
                vcb[wrd] = len(vcb)

    return feats, wrds, vcb


def lookup_numeric_labels(word_list, voc):
    # maps each unqiue word to a unique integer label
    labels = []
    for word in word_list:
        label = voc[word]
        labels.append([label])
    return np.array(labels)


if __name__ == '__main__':
    langs = sys.argv[1:3]
    lang1, lang2 = langs[0], langs[1]
    print(lang1, lang2)
    features, words, vocab = read_file('vectors6.' + lang1 + '.' + lang2)  # change input file here
    num_classes = len(vocab)    
    print('vocab size: ' + str(num_classes))
    num_features = len(features[0])
    num_tokens = len(features)
    batch_size = 256

    # graph input
    x = tf.placeholder(tf.float32, [None, num_features], name='x')
    y_true = tf.placeholder(tf.int32, [None, 1], name='y_true')

    # model (Multinomial Logistic Regression/Log-Linear model)
    W = tf.Variable(tf.zeros([num_features, num_classes]), name='W')
    b = tf.Variable(tf.zeros([num_classes]), name='b')
    y = tf.add(tf.matmul(x, W), b)

    # training
    # sparse loss function used to speed up computation
    # using unique labels is a lot less to hold in memory than one-hot vectors of vocab size dimensions
    loss_function = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.reshape(y_true, [-1]))
    learning_rate = 0.001
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss_function)

    # launch session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        old_loss = 1.0
        tol = 0.01
        max_iter = 100
        for j in range(max_iter):
            start = time.time()
            combined = list(zip(features, words))
            random.shuffle(combined)
            total_loss = 0.0
            for batch in chunks(combined, batch_size):
                batch_x, batch_y_words = zip(*batch)  # "unzip"
                batch_y_true = lookup_numeric_labels(batch_y_words, vocab)
                _, l = sess.run([optimizer, loss_function], feed_dict={x: np.array(batch_x), y_true: batch_y_true})
                current_loss = np.sum(l)
                assert not np.isnan(current_loss), 'Model diverged with loss = NaN'
                total_loss += current_loss

            print('after epoch:', j+1)
            print('W: ', sess.run(tf.reduce_mean(W, 1)))
            avg_loss = total_loss / num_tokens
            print('average loss:', avg_loss)
            perplexity = np.exp(avg_loss)  # TensorFlow's cross entropy is calculated with natural log
            print('perplexity:', perplexity)
            print('time:', time.time() - start)
            if np.abs(1.0 - avg_loss / old_loss) < tol:
                print('model converged')
                break
            old_loss = avg_loss