import collections
import glob
import os
import pickle
import sys

import numpy
import tensorflow as tf
import re  # regular expression operations

'''Return all text file in the given directory'''
def GetInputFiles():
    # print(glob.glob(os.path.join(sys.argv[1], '*/*/*/*.txt')))
    return glob.glob(os.path.join(sys.argv[1], '*/*/*/*.txt'))

""" DEBUG TIPS: """
VARS = {}

# vocab = {word 1: count 1}
VOCABULARY = collections.Counter()
stopWords = ["each","has", "had", "having", "do", "does", "did", "doing", "few", "more", "most", "other", "some", "such", "no","about", "against", "between", "into", "through", "during", "before","i", "me", "my", "myself", "we", "our", "ours","and", "but", "if",  "so", "than", "too", "very", "s", "t", "can", "will", "just", "or", "because", "as", "until", "while", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she",  "of", "at", "by", "for", "with",  "after", "above","her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "have",  "a", "an", "the",  "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both",  "nor",  "am", "is", "are", "was", "were", "be", "been", "being","not", "only", "own", "same","don", "should", "now"]

# remove words from stopWords
def removeStops(tokenList):
    output = []
    # loop through each token to see if it is in the drop list
    for token in tokenList:
        if token in stopWords:
            # do nothing
            continue
        else:
            # return token not in stopWords
            output.append(token)
    return output

# Function to remove punctuation from a token list []
def removePunctuations(tokenList):
    output = []
    # loop through each token
    for token in tokenList:
        # to do a regex substitution: remove all except letter, and inherent white spaces
        token = re.sub('[^a-zA-Z0-9\n\-]', '', token)
        output.append(token)
    return output

# Function to remove short words from a token list []
def removeShortTokens(tokenList):
    output = []
    # loop through each token
    for token in tokenList:
        # to do a regex substitution: remove all except letter, number and inherent white spaces
        if len(token) >= 2.0:
            output.append(token)
    return output

# ** TASK 1.
def Tokenize(comment):
    # split into words by white space
    tokens = re.split('[^a-zA-Z]', comment)

    # remove punctuations of each words
    # tokens = removePunctuations(tokens)

    # remove stop word
    # tokens = removeStops(tokens)

    # put them to lower case
    tokens = [token.lower() for token in tokens]

    # # remove punctuation from each word
    # table = str.maketrans('', '', string.punctuation)
    # strippedTokens = [w.translate(table) for w in tokens]

    # remove short word
    finalTokens = removeShortTokens(tokens)
    # print(finalTokens[:100])
    return finalTokens



# ** TASK 2.
def FirstLayer(net, l2_reg_val, is_training):
    """First layer of the neural network.

    Args:
      net: 2D tensor (batch-size, number of vocabulary tokens),
      l2_reg_val: float -- regularization coefficient.
      is_training: boolean tensor.A

    Returns:
      2D tensor (batch-size, 40), where 40 is the hidden dimensionality.??
    """

    # To do:
    # replace RELU with tanh
    # remove bias vector
    # Replace the L2-regularization of fully connected with manual regularization.
    # Preprocess the layer input by passing
    # Add Batch Normalization.

    global VARS
    ## Specify regularizer
    # Returns a function that can be used to apply L2 regularization to weights.
    # l2_reg = tf.contrib.layers.l2_regularizer(l2_reg_val)
    VARS["a"] = net  # X
    # print("first net: ", VARS["a"])

    ## Normalization row-wise for each data row with L2 norm
    net = tf.nn.l2_normalize(net, axis=1)
    VARS["b"] =net  # X/Xnorm
    # print("2nd net after l2 normalize: ", VARS["b"])

    ## Adds a fully connected layer: with input,
    # Output: weight matrix net

    net = tf.contrib.layers.fully_connected(
        net, 40, activation_fn=None, weights_regularizer=None, biases_initializer=None)
    VARS["c"] = net  # XY/Xnorm
    # print("third net after adding layer: ", VARS["c"])


    ## Because net = X*Y so Y = (X)^-1 * net
    loss_reg = l2_reg_val * net ** 2.0
    tf.losses.add_loss(tf.reduce_sum(loss_reg), loss_collection=tf.GraphKeys.REGULARIZATION_LOSSES)


    ## Add batch norm
    net = tf.contrib.layers.batch_norm(net, is_training=is_training)
    VARS["d"] = net
    # print("4th net after batch norm ", VARS["d"])

    ## Activation:??
    net = tf.math.tanh(net)
    VARS["e"] = net
    # print("5th net after tanh activation ", VARS["e"])

    return net


# ** TASK 2 ** BONUS part 1 ??
def EmbeddingL2RegularizationUpdate(embedding_variable, net_input, learn_rate, l2_reg_val):
    """Accepts tf.Variable, tensor (batch_size, vocab size), regularization coef.
    Returns tf op that applies one regularization step on embedding_variable."""
    # TODO(student): Change this to something useful. Currently, this is a no-op.
    # normalized_input = tf.nn.l2_normalize(net_input, axis=1)
    # # update_diff = learn_rate * (2*l2_reg_val * tf.matmul(embedding_variable, normalized_input))
    # return embedding_variable.assign(embedding_variable - learn_rate * (2*l2_reg_val * tf.matmul(normalized_input, embedding_variable)))
    return embedding_variable.assign(embedding_variable)


# ** TASK 2 ** BONUS part 2
def EmbeddingL1RegularizationUpdate(embedding_variable, net_input, learn_rate, l1_reg_val):
    """Accepts tf.Variable, tensor (batch_size, vocab size), regularization coef.
    Returns tf op that applies one regularization step on embedding_variable."""
    # TODO(student): Change this to something useful. Currently, this is a no-op.
    # normalized_input = tf.nn.l2_normalize(net_input, axis=1)
    # R = l1_reg_val * tf.norm(tf.matmul(normalized_input, embedding_variable), ord='1')
    # return embedding_variable.assign(embedding_variable - learn_rate * tf.gradients(R, embedding_variable))
    return embedding_variable.assign(embedding_variable)


# ** TASK 3
def SparseDropout(slice_x, keep_prob=0.5):
    """Sets random (1 - keep_prob) non-zero elements of slice_x to zero.

    Args:
      slice_x: 2D numpy array (batch_size, vocab_size)

    Returns:
      2D numpy array (batch_size, vocab_size)
    """
    # Get indices of non-zero elements:
    i, j = numpy.nonzero(slice_x)

    # Get random indices to set to zero
    indices = numpy.random.choice(len(i), int(numpy.floor((1-keep_prob) * len(i))), replace=False)

    # set the non-zero values at these random indices to zero
    slice_x[i[indices], j[indices]] =0

    return slice_x


# ** TASK 4
# TODO(student): YOU MUST SET THIS TO GET CREDIT.
# You should set it to tf.Variable of shape (vocabulary, 40).
EMBEDDING_VAR = tf.Variable(tf.zeros([len(VOCABULARY), 40]), dtype=tf.float32)


# ** TASK 5
# This is called automatically by VisualizeTSNE.
#  t-distributed stochastic neighbor embedding
def ComputeTSNE(embedding_matrix):
    """Projects embeddings onto 2D by computing tSNE.

    Args:
      embedding_matrix: numpy array of size (vocabulary, 40)

    Returns:
      numpy array of size (vocabulary, 2)
    """
    from sklearn.manifold import TSNE
    embedding_matrix_new = TSNE(n_components=2).fit_transform(embedding_matrix)
    # print("embedding matrix: ", embedding_matrix_new[:, :2])
    return embedding_matrix_new[:, :2]


# ** TASK 5
# This should save a PDF of the embeddings. This is the *only* function marked
# marked with "** TASK" that will NOT be automatically invoked by our grading
# script (it will be "stubbed-out", by monkey-patching). You must run this
# function on your own, save the PDF produced by it, and place it in your
# submission directory with name 'tsne_embeds.pdf'.
def VisualizeTSNE(sess):
    if EMBEDDING_VAR is None:
        print('Cannot visualize embeddings. EMBEDDING_VAR is not set')
        return
    embedding_mat = sess.run(EMBEDDING_VAR)
    tsne_embeddings = ComputeTSNE(embedding_mat)
    # print("tsne embeddings: ", tsne_embeddings)

    class_to_words = {
        'positive': [
            'relaxing', 'upscale', 'luxury', 'luxurious', 'recommend', 'relax',
            'choice', 'best', 'pleasant', 'incredible', 'magnificent',
            'superb', 'perfect', 'fantastic', 'polite', 'gorgeous', 'beautiful',
            'elegant', 'spacious'
        ],
        'location': [
            'avenue', 'block', 'blocks', 'doorman', 'windows', 'concierge', 'living'
        ],
        'furniture': [
            'bedroom', 'floor', 'table', 'coffee', 'window', 'bathroom', 'bath',
            'pillow', 'couch'
        ],
        'negative': [
            'dirty', 'rude', 'uncomfortable', 'unfortunately', 'ridiculous',
            'disappointment', 'terrible', 'worst', 'mediocre'
        ]
    }


    # TODO(student): Visualize scatter plot of tsne_embeddings, showing only words
    # listed in class_to_words. Words under the same class must be visualized with
    # the same color. Plot both the word text and the tSNE coordinates.

    # print("Term index: ", TERM_INDEX)
    # print("Vocabulary: ", VOCABULARY)

    # need to extract 2 list:
    # labels: all the words in class_to_words
    # sub_tsne_embeddings: part of tsne_embeddings corresponding to those words in labels
    selected_words = []
    selected_tsne = []
    selected_classes = []

    # ?? SUPER SLOW
    for cluster in class_to_words:
        for word in class_to_words[cluster]:
            # add words and its corresponding tsne
            selected_words.append(word)
            selected_classes.append(cluster)
            selected_tsne.append(tsne_embeddings[TERM_INDEX[word]])

    # print("Classes: ", selected_classes)

    import matplotlib.pyplot as plt

    x = []
    y = []
    for value in selected_tsne:
        x.append(value[0])
        y.append(value[1])


    # fig, ax = plt.subplots()
    # # f=plt.figure(figsize=(16, 16))
    # df = pd.DataFrame(dict(x=x, y=y, classes=selected_classes), index=selected_words)
    # df.plot('x', 'y', kind='scatter', ax=ax)
    # for k, v in df.iterrows():
    #     ax.annotate(k, v)
    #
    # # plt.annotate(selected_words[i],
    # #              xy=(x[i], y[i]),
    # #              xytext=(5, 2),
    # #              textcoords='offset points',
    # #              ha='right',
    # #              va='bottom')
    # sns.lmplot('x', 'y', data=df, hue='classes', fit_reg=False)
    # plt.show()
    # plt.savefig("tsne_embeds.pdf", bbox_inches='tight')
    # plt.close()

    colors = {'positive':'red', 'negative':'green', 'furniture':'blue', 'location':'purple'}

    plt.figure(figsize=(16, 16))
    for i in range(len(x)):
        plt.scatter(x[i], y[i], color=colors[selected_classes[i]])
        plt.annotate(selected_words[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.title('Selected Word Embedding TSNE')
    # plt.legend(loc='best')

    # save the plot to pdf file
    plt.savefig("tsne_embeds.pdf", bbox_inches='tight')
    print('visualization should be saved now')

    plt.show()
    print('visualization should generate now')

    plt.close()

CACHE = {}

'''Read and tokenize a file with fileName'''
def ReadAndTokenize(filename):
    """return dict containing of terms to frequency."""
    global CACHE
    global VOCABULARY

    # search if file is already in CACHE
    if filename in CACHE:
        return CACHE[filename]

    # open content of the file
    comment = open(filename).read()

    # tokenize into list of words
    words = Tokenize(comment)

    # counting appearance of each word of vocab in words
    terms = collections.Counter()

    # loop through each word
    for w in words:
        # update the count and vocab
        VOCABULARY[w] += 1
        terms[w] += 1

    # update CACHE
    CACHE[filename] = terms
    return terms

# This global variable is used to track {word1: index1}
TERM_INDEX = None

'''Part of word embedding process'''
### input: X: [terms1, terms2,...] with terms1 representing doc1 = {word1:count1, word2:count2]
### output: X_matrix: #doc*#features
def MakeDesignMatrix(x):
    global TERM_INDEX
    if TERM_INDEX is None:
        print('Total words: %i' % len(VOCABULARY.values()))

        # Returns the q-th percentile(s) of the array elements.
        # min_count is more like median count because of sparse data??
        min_count, max_count = numpy.percentile(list(VOCABULARY.values()), [50.0, 99.8])
        # print("min_count ", min_count)
        # print("max_count ", max_count)

        # only perform embedding when word frequency reaches certain threshold >50th percentile
        TERM_INDEX = {}
        for term, count in VOCABULARY.items():
            if count > min_count and count <= max_count:
                # add terms sequentially with their index
                idx = len(TERM_INDEX)
                TERM_INDEX[term] = idx

    # initiate x_matrix
    x_matrix = numpy.zeros(shape=[len(x), len(TERM_INDEX)], dtype='float32')

    # loop through x with x = [doc1,doc2...] and doc1={token1:count1, token2:count2,...}
    for i, item in enumerate(x):
        # loop through each token and its count
        for term, count in item.items():
            if term not in TERM_INDEX:
                continue

            # get the necessary index of each term
            j = TERM_INDEX[term]

            # update the count in x_matrix
            x_matrix[i, j] = count  # 1.0  # Try count or log(1+count)
    return x_matrix

'''Construct train and test data from all text files and make matrices'''
def GetDataset():
    """Returns numpy arrays of training and testing data."""
    x_train = []
    x_test = []
    y_train = []
    y_test = []

    classes1 = set()
    classes2 = set()

    # loop through text files from the directory
    for f in GetInputFiles():
        # print(f)
        # print(f.split('\\')[-4:])

        # extract each class (Truthful/Deceptive, Positive/Negative), data fold and file name
        class1, class2, fold, fname = f.split('\\')[-4:]
        classes1.add(class1)
        classes2.add(class2)
        class1 = class1.split('_')[0] #??
        class2 = class2.split('_')[0]

        # read and tokenize each text file
        x = ReadAndTokenize(f)

        # y is a list [1,1] for positive and truthful...
        y = [int(class1 == 'positive'), int(class2 == 'truthful')]

        # save fold 4 for testing
        if fold == 'fold4':
            x_test.append(x)
            y_test.append(y)
        # add the rest to train data
        else:
            x_train.append(x)
            y_train.append(y)

    ### Make numpy arrays: transform train and test data to matrices
    x_test = MakeDesignMatrix(x_test) # numDocs*numFeatures
    x_train = MakeDesignMatrix(x_train)
    y_test = numpy.array(y_test, dtype='float32') # numDocs*numClasses
    y_train = numpy.array(y_train, dtype='float32')

    # combine to dataset as pickle
    dataset = (x_train, y_train, x_test, y_test)

    # write binary this dataset
    with open('dataset.pkl', 'wb') as fout:
        pickle.dump(dataset, fout)
    return dataset


'''print out evaluation results'''
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

''' Construct neural network'''
def BuildInferenceNetwork(x, l2_reg_val, is_training):
    """From a tensor x, runs the neural network forward to compute outputs.
    This essentially instantiates the network and all its parameters.

    Args:
      x: Tensor of shape (batch_size, vocab size) which contains a sparse matrix
         where each row is a training example and containing counts of words
         in the document that are known by the vocabulary.

    Returns:
      Tensor of shape (batch_size, 2) where the 2-columns represent class
      memberships: one column discriminates between (negative and positive) and
      the other discriminates between (deceptive and truthful).
    """
    global EMBEDDING_VAR
    EMBEDDING_VAR = None # ** TASK 4: Move and set appropriately.
    # print("Embedding var: ", EMBEDDING_VAR)

    ## Build layers starting from input.
    net = x

    # get L2 regularizer value
    l2_reg = tf.contrib.layers.l2_regularizer(l2_reg_val)

    # print("trainable variables before first layer: ", tf.trainable_variables())

    ## First Layer
    net = FirstLayer(net, l2_reg_val, is_training)
    EMBEDDING_VAR = [v for v in tf.global_variables() if v.name == "fully_connected/weights:0"][0]
    # print("Embedding var after first layer: ", EMBEDDING_VAR)

    # print("trainable variables after first layer: ", tf.trainable_variables())

    ## Second Layer.
    # create a fully connected layer:
    net = tf.contrib.layers.fully_connected(
        net, 10, activation_fn=None, weights_regularizer=l2_reg)
    EMBEDDING_VAR = [v for v in tf.global_variables() if v.name == "fully_connected/weights:0"][0]
    # print("Embedding var after second layer: ", EMBEDDING_VAR)

    # perform dropout
    net = tf.contrib.layers.dropout(net, keep_prob=0.5, is_training=is_training)
    EMBEDDING_VAR = [v for v in tf.global_variables() if v.name == "fully_connected/weights:0"][0]
    # print("Embedding var after DROPOUT: ", EMBEDDING_VAR)

    # perform activation function
    net = tf.nn.relu(net)
    EMBEDDING_VAR = [v for v in tf.global_variables() if v.name == "fully_connected/weights:0"][0]
    # print("Embedding var after second layer's activation: ", EMBEDDING_VAR)

    ## Third Layer
    net = tf.contrib.layers.fully_connected(
        net, 2, activation_fn=None, weights_regularizer=l2_reg)
    EMBEDDING_VAR = [v for v in tf.global_variables() if v.name == "fully_connected/weights:0"][0]
    # print("Embedding var after third and final layer: ", EMBEDDING_VAR)

    return net

'''MAIN with argument'''
def main(argv):
    ######### Read dataset
    x_train, y_train, x_test, y_test = GetDataset()

    ######### Neural Network Model
    # set placeholders the same size with test data: X is a matrix of input features (# docs* features)
    x = tf.placeholder(tf.float32, [None, x_test.shape[1]], name='x')

    # target matrix Y: # docs * numClassses
    y = tf.placeholder(tf.float32, [None, y_test.shape[1]], name='y')

    is_training = tf.placeholder(tf.bool, [])  #boolean tensor

    # Co-efficient for L2 regularization (lambda)
    l2_reg_val = 1e-6

    # Build inference network based on training data, regularization coefficient and boolean is_training
    net = BuildInferenceNetwork(x, l2_reg_val, is_training)

    ######### Loss Function:
    tf.losses.sigmoid_cross_entropy(multi_class_labels=y, logits=net)

    ######### Training Algorithm
    # learning rate
    learning_rate = tf.placeholder_with_default(
        numpy.array(0.01, dtype='float32'), shape=[], name='learn_rate')

    # optimizer is gradient descent optimizer
    opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

    # training optimizer
    train_op = tf.contrib.training.create_train_op(tf.losses.get_total_loss(), opt)

    # Run a session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())


    # FOR DEBUGGING
    global VARS
    # import IPython;  IPython.embed()

    # Function to evaluate on a batch of sample
    def evaluate(batch_x=x_test, batch_y=y_test):
        probs = sess.run(net, {x: batch_x, is_training: False})
        print_f1_measures(probs, batch_y)

    # Function to learn on a batch of sample with learning rate lr
    def batch_step(batch_x, batch_y, lr):
        sess.run(train_op, {
            x: batch_x,
            y: batch_y,
            is_training: True, learning_rate: lr,
        })

    # randomly slice training data based on batch_size and lr
    def step(lr=0.01, batch_size=100):
        indices = numpy.random.permutation(x_train.shape[0])
        for si in range(0, x_train.shape[0], batch_size):
            se = min(si + batch_size, x_train.shape[0])
            slice_x = x_train[indices[si:se]] + 0  # + 0 to copy slice

            # perform sparse dropout
            slice_x = SparseDropout(slice_x)

            # Get a batch of training data
            batch_step(slice_x, y_train[indices[si:se]], lr)

    lr = 0.05
    print('Training model ... ')
    for j in range(300): step(lr)
    for j in range(300): step(lr / 2)
    for j in range(300): step(lr / 4)
    print('Results from training:')
    evaluate()


    # Visualize results
    VisualizeTSNE(sess)

    #### Save parameters:
    # var_dict = {v.name: v for v in tf.global_variables()}
    # pickle.dump(sess.run(var_dict), open('trained_vars.pkl', 'w'))

    #### Restore parameters for prediction
    # var_values = pickle.load(open('trained_vars.pkl'))
    # assign_ops = [v.assign(var_values[v.name]) for v in tf.global_variables()]
    # sess.run(assign_ops)


if __name__ == '__main__':
    # set random seed
    tf.random.set_random_seed(0)

    # run main
    main([])