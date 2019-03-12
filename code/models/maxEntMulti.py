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
    vcb = {'UNK': 0}
    
    with open(file) as inpt:
        for line in inpt:  # for looping over whole dataset
        #for line in itertools.islice(inpt, 49999):  # for taking first n lines of dataset (for development)
            split = line.strip().split()
            feat_vec = [float(val) for val in split[:3]] + [float(val) for val in split[4:-1]]
            feats.append(feat_vec)
            wrd = split[-1]
            wrds.append(wrd)
            if wrd not in vcb:
                vcb[wrd] = len(vcb)

    return feats, wrds, vcb


def lookup_numeric_labels(word_list, voc):
    # maps each unique word to a unique integer label
    return np.array([[voc[word]] if word in voc else voc['UNK'] for word in word_list])


if __name__ == '__main__':
    langs = sys.argv[1:3]
    lang1, lang2 = langs[0], langs[1]
    print(lang1, lang2)
    features, words, vocab = read_file('data/vectors6.' + lang1 + '.' + lang2)
    #test_x, test_words, test_vocab = read_file('test_vectors6.' + lang1 + '.' + lang2)
    #test_y_true = lookup_numeric_labels(test_words, vocab)
    num_classes = len(vocab)
    print('vocab size: ' + str(num_classes))
    num_features = len(features[0])
    num_tokens = len(features)
    batch_size = 256
    num_gpus = 4


    def parallelize(fn, num_gpus, **kwargs):
    	# input data comes in as a multiple of num_gpus
        input_split = {}
        for k, v in kwargs.items():
            input_split[k] = []
            for i in range(num_gpus):
            	# slice up the data into equal sized pieces to send to each gpu
                shape = tf.shape(v)
                size = tf.concat([shape[:1] // num_gpus, shape[1:]], axis=0)
                stride = tf.concat([shape[:1] // num_gpus, shape[1:]*0], axis=0)
                start = stride * i
                input_split[k].append(tf.slice(v, start, size))

        output_split = []
        for i in range(num_gpus):
            with tf.device(tf.DeviceSpec(device_type="GPU", device_index=i)):
                with tf.variable_scope(tf.get_variable_scope(), reuse=i > 0):
                	# send data through model one gpu at a time
                    output_split.append(fn(**{k : v[i] for k, v in input_split.items()}))
        
        # return all the losses returned from each gpu
        return tf.concat(output_split, axis=0)


    def maxEnt_model(ex, y_tru):
    	# draw graph (Multinomial Logistic Regression/Log-Linear model)
        # W = tf.Variable(tf.zeros([num_features, num_classes]), name='W')
        W = tf.get_variable('W', [num_features, num_classes], initializer=tf.contrib.layers.xavier_initializer())  # based on Glorot and Bengio (2010)
        b = tf.Variable(tf.zeros([num_classes]), name='b')
        # calculate log probs based on Beta_0 + Beta_1 * x
        y = tf.add(tf.matmul(ex, W), b)
        # sparse loss function used to speed up computation
        # using unique labels is a lot less to hold in memory than one-hot vectors of vocab size dimensions
        loss_func = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.reshape(y_tru, [-1]))
        return loss_func

    # graph input
    x = tf.placeholder(tf.float32, [None, num_features], name='x')
    y_true = tf.placeholder(tf.int32, [None, 1], name='y_true')

    loss_function = parallelize(maxEnt_model, num_gpus, ex=x, y_tru=y_true)
    learning_rate = 0.001
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss_function, colocate_gradients_with_ops=True)

    # launch session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        old_PP = 1.0
        tol = 0.05
        max_iter = 100
        for j in range(max_iter):
            start = time.time()
            combined = list(zip(features, words))
            random.shuffle(combined)
            total_loss = 0.0
            for batch in chunks(combined, batch_size * num_gpus):
                batch_x, batch_y_words = zip(*batch)  # "unzip"
                batch_y_true = lookup_numeric_labels(batch_y_words, vocab)
                _, l = sess.run([optimizer, loss_function], feed_dict={x: np.array(batch_x), y_true: batch_y_true})
                batch_loss = np.sum(l)
                assert not np.isnan(batch_loss), 'Model diverged with loss = NaN'
                total_loss += batch_loss

            print('after epoch:', j+1)
            print('W:', np.mean(sess.run(tf.contrib.framework.get_variables_by_name('W'))[0], axis=1))
            avg_loss = total_loss / num_tokens
            print('average loss:', avg_loss)
            perplexity = np.exp(avg_loss)  # TensorFlow's cross entropy is calculated with natural log
            print('perplexity:', perplexity)
            print('time:', time.time() - start)
            if np.abs(1.0 - perplexity / old_PP) < tol:
                print('model converged')
                break
            old_PP = perplexity
