"""
DOCUMENT CLASSIFICATION USING BAREBONES TENSORFLOW
"""

"""
General gist.
    Given (X) and (Y): First is a matrix of input features (# docs x features) and second is target matrix (# of docs x number of classes)
    Start with input matrix (X)
    Apply a series of transformations to the matrix.
    The last transformation produces h(X)
    Add a loss term that encourages h(X) to be as similar as possible to matrix (Y)
"""

"""
SAVING MODEL
You can save the parameters like:
import picklevar_dict = {v.name: v for v in tf.global_variables()}pickle.dump(sess.run(var_dict), open('trained_vars.pkl', 'w'))

And restore like:
import picklevar_values = pickle.load(open('trained_vars.pkl'))assign_ops = [v.assign(var_values[v.name]) for v in tf.global_variables()]sess.run(assign_ops)
"""
import collections
import glob
import os
import pickle
import re
import sys

from absl import app
from absl import flags
import numpy
import tensorflow as tf

flags.DEFINE_integer('layers', 1, 'Number of Neural Net Layers.')

FLAGS = flags.FLAGS

all_files = glob.glob(os.path.join(sys.argv[1], '*/*/*/*.txt'))

VOCABULARY = collections.Counter()

CACHE = {}


def ReadAndTokenize(filename):
    """return dict containing of terms to frequency."""
    global CACHE
    global VOCABULARY
    if filename in CACHE:
        return CACHE[filename]
    words = open(filename).read().split()
    terms = collections.Counter()
    for w in words:
        w = w.lower()
        VOCABULARY[w] += 1
        terms[w] += 1

    CACHE[filename] = terms
    return terms


TERM_INDEX = None


def MakeDesignMatrix(x):
    global TERM_INDEX
    if TERM_INDEX is None:
        min_count, max_count = numpy.percentile(list(VOCABULARY.values()), [50.0, 99.5])
        TERM_INDEX = {}
        for term, count in VOCABULARY.items():
            if count >= min_count and count <= max_count:
                idx = len(TERM_INDEX)
                TERM_INDEX[term] = idx
    #
    x_matrix = numpy.zeros(shape=[len(x), len(TERM_INDEX)], dtype='float32')
    for i, item in enumerate(x):
        for term, count in item.items():
            if term not in TERM_INDEX:
                continue
            j = TERM_INDEX[term]
            x_matrix[i, j] = count  # 1.0  # Try count or log(1+count)
    return x_matrix


def GetDataset():
    """Returns numpy arrays of training and testing data."""
    # if os.path.exists('dataset.pkl'):
    #  return pickle.load(open('dataset.pkl', 'rb'))
    x_train = []
    x_test = []
    y_train = []
    y_test = []

    classes1 = set()
    classes2 = set()
    for f in all_files:
        # print(f)
        # print(f.split('\\'))
        class1, class2, fold, fname = f.split('\\')[-4:]
        classes1.add(class1)
        classes2.add(class2)
        class1 = class1.split('_')[0]
        class2 = class2.split('_')[0]

        x = ReadAndTokenize(f)
        y = [int(class1 == 'positive'), int(class2 == 'truthful')]
        if fold == 'fold4':
            x_test.append(x)
            y_test.append(y)
        else:
            x_train.append(x)
            y_train.append(y)

    ### Make numpy arrays.
    x_test = MakeDesignMatrix(x_test)
    x_train = MakeDesignMatrix(x_train)
    y_test = numpy.array(y_test, dtype='float32')
    y_train = numpy.array(y_train, dtype='float32')

    dataset = (x_train, y_train, x_test, y_test)
    with open('dataset.pkl', 'wb') as fout:
        pickle.dump(dataset, fout)
    return dataset


def print_f1_measures(probs, y_test):
    y_test[:, 0] == 1  # Positive
    positive = {
        'tp': numpy.sum((probs[:, 0] > 0)[numpy.nonzero(y_test[:, 0] == 1)[0]]),
        'fp': numpy.sum((probs[:, 0] > 0)[numpy.nonzero(y_test[:, 0] == 0)[0]]),
        'fn': numpy.sum((probs[:, 0] <= 0)[numpy.nonzero(y_test[:, 0] == 1)[0]]),
    }
    negative = {
        'tp': numpy.sum((probs[:, 0] <= 0)[numpy.nonzero(y_test[:, 0] == 0)[0]]),
        'fp': numpy.sum((probs[:, 0] <= 0)[numpy.nonzero(y_test[:, 0] == 1)[0]]),
        'fn': numpy.sum((probs[:, 0] > 0)[numpy.nonzero(y_test[:, 0] == 0)[0]]),
    }
    truthful = {
        'tp': numpy.sum((probs[:, 1] > 0)[numpy.nonzero(y_test[:, 1] == 1)[0]]),
        'fp': numpy.sum((probs[:, 1] > 0)[numpy.nonzero(y_test[:, 1] == 0)[0]]),
        'fn': numpy.sum((probs[:, 1] <= 0)[numpy.nonzero(y_test[:, 1] == 1)[0]]),
    }
    deceptive = {
        'tp': numpy.sum((probs[:, 1] <= 0)[numpy.nonzero(y_test[:, 1] == 0)[0]]),
        'fp': numpy.sum((probs[:, 1] <= 0)[numpy.nonzero(y_test[:, 1] == 1)[0]]),
        'fn': numpy.sum((probs[:, 1] > 0)[numpy.nonzero(y_test[:, 1] == 0)[0]]),
    }

    all_f1 = []
    for attribute_name, score in [('truthful', truthful),
                                  ('deceptive', deceptive),
                                  ('positive', positive),
                                  ('negative', negative)]:
        precision = float(score['tp']) / float(score['tp'] + score['fp'])
        recall = float(score['tp']) / float(score['tp'] + score['fn'])
        f1 = 2 * precision * recall / (precision + recall)
        all_f1.append(f1)
        print('{0:9} {1:.2f} {2:.2f} {3:.2f}'.format(attribute_name, precision, recall, f1))
    print('Mean F1: {0:.4f}'.format(float(sum(all_f1)) / len(all_f1)))


def main(argv):
    # import IPython; IPython.embed()
    ######### Read dataset
    x_train, y_train, x_test, y_test = GetDataset()

    ######### Neural Network Model
    x = tf.placeholder(tf.float32, [None, x_test.shape[1]], name='x')
    y = tf.placeholder(tf.float32, [None, y_test.shape[1]], name='y')
    is_training = tf.placeholder(tf.bool, [])

    l2_reg = tf.contrib.layers.l2_regularizer(1e-6)

    ## Build layers starting from input.
    net = x

    ## Hidden layer
    if FLAGS.layers >= 2:
        net = tf.contrib.layers.fully_connected(
            net, 40, activation_fn=None, weights_regularizer=l2_reg)
        net = tf.contrib.layers.dropout(net, keep_prob=0.3, is_training=is_training)
        net = tf.contrib.layers.batch_norm(net, is_training=is_training)
        net = tf.nn.relu(net)

    ## Hidden layer
    if FLAGS.layers >= 3:
        net = tf.contrib.layers.fully_connected(
            net, 10, activation_fn=None, weights_regularizer=l2_reg)
        net = tf.contrib.layers.dropout(net, keep_prob=0.5, is_training=is_training)
        net = tf.contrib.layers.batch_norm(net, is_training=is_training)
        net = tf.nn.relu(net)

    ## Output Layer.
    net = tf.contrib.layers.fully_connected(
        net, 2, activation_fn=None, weights_regularizer=l2_reg)

    ######### Loss Function
    tf.losses.sigmoid_cross_entropy(multi_class_labels=y, logits=net)

    ######### Training Algorithm
    learning_rate = tf.placeholder_with_default(
        numpy.array(0.01, dtype='float32'), shape=[], name='learn_rate')
    opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = tf.contrib.training.create_train_op(tf.losses.get_total_loss(), opt)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    def evaluate():
        probs = sess.run(net, {x: x_test, is_training: False})
        print_f1_measures(probs, y_test)

    def step(lr=0.01):
        batch_size = 200
        indices = numpy.random.permutation(x_train.shape[0])
        for si in range(0, x_train.shape[0], batch_size):
            se = min(si + batch_size, x_train.shape[0])
            sess.run(train_op, {
                is_training: True,
                learning_rate: lr,
                x: x_train[indices[si:se]],
                y: y_train[indices[si:se]],
            })

    lr = 0.01
    for j in range(100): step(lr)
    for j in range(100): step(lr / 2)
    for j in range(100): step(lr / 4)
    for j in range(200): step(lr / 8)
    for j in range(200): step(lr / 16)
    evaluate()


if __name__ == '__main__':
    tf.random.set_random_seed(0)
    app.run(main)
