import sys
import time
import pickle
import numpy
import tensorflow as tf


# maxLenSentence = 0


class DatasetReader(object):

    # TODO(student): You must implement this.
    @staticmethod
    def ReadFile(filename, term_index, tag_index):
        """Reads file into dataset, while populating term_index and tag_index.
        Args:
            filename: Path of text file containing sentences and tags. Each line is a
                sentence and each term is followed by "/tag". Note: some terms might
                have a "/" e.g. my/word/tag -- the term is "my/word" and the last "/"
                separates the tag.
            term_index: dictionary to be populated with every unique term (i.e. before
                the last "/") to point to an integer. All integers must be utilized from
                0 to number of unique terms - 1, without any gaps nor repetitions.
            tag_index: same as term_index, but for tags.
        the _index dictionaries are guaranteed to have no gaps when the method is
        called i.e. all integers in [0, len(*_index)-1] will be used as values.
        You must preserve the no-gaps property!
        Return:
            The parsed file as a list of lists: [parsedLine1, parsedLine2, ...]
            each parsedLine is a list: [(termId1, tagId1), (termId2, tagId2), ...]
        """
        with open(filename, 'r') as f:
            lines = f.readlines()
        f.close()

        # Pass1 - update term_index and tag_index
        # prev_term_index = 0
        # prev_tag_index = -1
        # Pass1 - update term_index and tag_index
        if len(term_index) > 0:
            max_value = max(term_index.values())
            prev_term_index = max_value
        else:
            prev_term_index = -1

        if len(tag_index) > 0:
            max_value = max(tag_index.values())
            prev_tag_index = max_value
        else:
            prev_tag_index = -1

        # global maxLenSentence
        for line in lines:
            split_line = line.split(' ')
            # maxLenSentence = max(maxLenSentence,len(split_line))

            for data in split_line:
                # print(data)
                data = data.strip()
                indexOfSeparator = data.rindex('/')
                word = data[:indexOfSeparator]
                tag = data[indexOfSeparator + 1:]
                if word not in term_index:
                    term_index[word] = prev_term_index + 1
                    prev_term_index += 1
                if tag not in tag_index:
                    tag_index[tag] = prev_tag_index + 1
                    prev_tag_index += 1

        # print("len of term_index")
        # print(len(term_index))
        # print(term_index)
        # print("len of tag_index")
        # print(len(tag_index))
        # print(tag_index)

        # Pass2 - Prepare return type

        list1 = []
        for line in lines:
            list2 = []
            for data in line.split(' '):
                data = data.strip()
                indexOfSeparator = data.rindex('/')
                word = data[:indexOfSeparator]
                tag = data[indexOfSeparator + 1:]
                tup = (term_index[word], tag_index[tag])
                list2.append(tup)
            list1.append(list2)

        # print(len(list1))
        # print("list1[3]")
        # print(list1[3])
        return list1
        # pass

        # pass

    # TODO(student): You must implement this.
    @staticmethod
    def BuildMatrices(dataset):
        """Converts dataset [returned by ReadFile] into numpy arrays for tags, terms, and lengths.
        Args:
            dataset: Returned by method ReadFile. It is a list (length N) of lists:
                [sentence1, sentence2, ...], where every sentence is a list:
                [(word1, tag1), (word2, tag2), ...], where every word and tag are integers.
        Returns:
            Tuple of 3 numpy arrays: (terms_matrix, tags_matrix, lengths_arr)
                terms_matrix: shape (N, T) int64 numpy array. Row i contains the word
                    indices in dataset[i].
                tags_matrix: shape (N, T) int64 numpy array. Row i contains the tag
                    indices in dataset[i].
                lengths: shape (N) int64 numpy array. Entry i contains the length of
                    sentence in dataset[i].
            T is the maximum length. For example, calling as:
                BuildMatrices([[(1,2), (4,10)], [(13, 20), (3, 6), (7, 8), (3, 20)]])
            i.e. with two sentences, first with length 2 and second with length 4,
            should return the tuple:
            (
                [[1, 4, 0, 0],    # Note: 0 padding.
                 [13, 3, 7, 3]],
                [[2, 10, 0, 0],   # Note: 0 padding.
                 [20, 6, 8, 20]],
                [2, 4]
            )
        """
        global maxLenSentence
        term_matrix = []
        tags_matrix = []
        length = []
        # main_matrix = []

        for list1 in dataset:
            length.append(len(list1))

        local_max_length = max(length)

        print("buildmatrices ")
        print(local_max_length)
        for list1 in dataset:
            succeeding_zero_count = local_max_length - len(list1)
            if (succeeding_zero_count < 0):
                print("succeeding_zero_count")
                print(succeeding_zero_count)

            lsTerm = numpy.array([termIndex[0] for termIndex in list1])
            lsTerm = numpy.pad(lsTerm, (0, succeeding_zero_count), 'constant', constant_values=(0, 0))
            lsTag = numpy.array([termIndex[1] for termIndex in list1])
            lsTag = numpy.pad(lsTag, (0, succeeding_zero_count), 'constant', constant_values=(0, 0))
            term_matrix.append(lsTerm)
            tags_matrix.append(lsTag)
            # length.append(len(list1))

        tup = (numpy.array(term_matrix), numpy.array(tags_matrix), numpy.array(length))
        return tup
        # pass

    @staticmethod
    def ReadData(train_filename, test_filename=None):
        """Returns numpy arrays and indices for train (and optionally test) data.
        NOTE: Please do not change this method. The grader will use an identitical
        copy of this method (if you change this, your offline testing will no longer
        match the grader).
        Args:
            train_filename: .txt path containing training data, one line per sentence.
                The data must be tagged (i.e. "word1/tag1 word2/tag2 ...").
            test_filename: Optional .txt path containing test data.
        Returns:
            A tuple of 3-elements or 4-elements, the later iff test_filename is given.
            The first 2 elements are term_index and tag_index, which are dictionaries,
            respectively, from term to integer ID and from tag to integer ID. The int
            IDs are used in the numpy matrices.
            The 3rd element is a tuple itself, consisting of 3 numpy arrsys:
                - train_terms: numpy int matrix.
                - train_tags: numpy int matrix.
                - train_lengths: numpy int vector.
                These 3 are identical to what is returned by BuildMatrices().
            The 4th element is a tuple of 3 elements as above, but the data is
            extracted from test_filename.
        """
        term_index = {'__oov__': 0}  # Out-of-vocab is term 0.
        tag_index = {}

        train_data = DatasetReader.ReadFile(train_filename, term_index, tag_index)
        train_terms, train_tags, train_lengths = DatasetReader.BuildMatrices(train_data)

        if test_filename:
            test_data = DatasetReader.ReadFile(test_filename, term_index, tag_index)
            test_terms, test_tags, test_lengths = DatasetReader.BuildMatrices(test_data)

            if test_tags.shape[1] < train_tags.shape[1]:
                diff = train_tags.shape[1] - test_tags.shape[1]
                zero_pad = numpy.zeros(shape=(test_tags.shape[0], diff), dtype='int64')
                test_terms = numpy.concatenate([test_terms, zero_pad], axis=1)
                test_tags = numpy.concatenate([test_tags, zero_pad], axis=1)
            elif test_tags.shape[1] > train_tags.shape[1]:
                diff = test_tags.shape[1] - train_tags.shape[1]
                zero_pad = numpy.zeros(shape=(train_tags.shape[0], diff), dtype='int64')
                train_terms = numpy.concatenate([train_terms, zero_pad], axis=1)
                train_tags = numpy.concatenate([train_tags, zero_pad], axis=1)

            return (term_index, tag_index,
                    (train_terms, train_tags, train_lengths),
                    (test_terms, test_tags, test_lengths))
        else:
            return term_index, tag_index, (train_terms, train_tags, train_lengths)


class SequenceModel(object):

    def __init__(self, max_length=310, num_terms=1000, num_tags=40):
        """Constructor. You can add code but do not remove any code.
        The arguments are arbitrary: when you are training on your own, PLEASE set
        them to the correct values (e.g. from main()).
        Args:
            max_lengths: maximum possible sentence length.
            num_terms: the vocabulary size (number of terms).
            num_tags: the size of the output space (number of tags).
        You will be passed these arguments by the grader script.
        """
        self.max_length = max_length
        self.num_terms = num_terms
        self.num_tags = num_tags
        self.x = tf.placeholder(tf.int64, [None, self.max_length], 'X')
        self.lengths = tf.placeholder(tf.int32, [None], 'lengths')

        # self.sess.run(tf.global_variables_initializer())

    # TODO(student): You must implement this.
    def lengths_vector_to_binary_matrix(self, length_vector):
        """Returns a binary mask (as float32 tensor) from (vector) int64 tensor.
        Specifically, the return matrix B will have the following:
            B[i, :lengths[i]] = 1 and B[i, lengths[i]:] = 0 for each i.
        However, since we are using tensorflow rather than numpy in this function,
        you cannot set the range as described.
        """
        # return tf.ones([tf.shape(length_vector), self.max_length], dtype=tf.float32)
        return tf.sequence_mask(length_vector, self.max_length, dtype=tf.float32)

    # TODO(student): You must implement this.
    def save_model(self, filename):
        """Saves model to a file."""
        # pass
        var_dict = {v.name: v for v in tf.global_variables()}
        pickle.dump(self.sess.run(var_dict), open('trained_vars.pkl', 'w'))

    # TODO(student): You must implement this.
    def load_model(self, filename):
        """Loads model from a file."""
        # pass
        var_values = pickle.load(open('trained_vars.pkl'))
        assign_ops = [v.assign(var_values[v.name]) for v in tf.global_variables()]
        self.sess.run(assign_ops)

    # TODO(student): You must implement this.
    def build_inference(self):
        """Build the expression from (self.x, self.lengths) to (self.logits).
        Please do not change or override self.x nor self.lengths in this function.
        Hint:
            - Use lengths_vector_to_binary_matrix
            - You might use tf.reshape, tf.cast, and/or tensor broadcasting.
        """
        # TODO(student): make logits an RNN on x.
        # self.logits = tf.zeros([tf.shape(self.x)[0], self.max_length, self.num_tags])
        state_size = 80
        # emb = tf.Variable(tf.zeros([self.num_terms,state_size]), tf.int32)
        emb = tf.get_variable('embeddings', shape=[self.num_terms, state_size])
        # self.sess.run(emb)
        xemb = tf.nn.embedding_lookup(emb, self.x)
        rnn_cell = tf.keras.layers.SimpleRNNCell(state_size)
        states = []
        cur_state = tf.zeros([1, state_size])
        for i in range(self.max_length):
            cur_state = rnn_cell(xemb[:, i, :], [cur_state])[0]
            states.append(cur_state)
        stacked_states = tf.stack(states, axis=1)

        # batch, max_len, 80
        # 80 -> num_tags
        # FC, dense layer - keras

        dense_layer = tf.keras.layers.Dense(self.num_tags, activation=None)
        self.logits = dense_layer(stacked_states)

    # TODO(student): You must implement this.
    def run_inference(self, terms, lengths):
        """Evaluates self.logits given self.x and self.lengths.
        Hint: This function is straight forward and you might find this code useful:
        # logits = session.run(self.logits, {self.x: terms, self.lengths: lengths})
        # return numpy.argmax(logits, axis=2)
        Args:
            terms: numpy int matrix, like terms_matrix made by BuildMatrices.
            lengths: numpy int vector, like lengths made by BuildMatrices.
        Returns:
            numpy int matrix of the predicted tags, with shape identical to the int
            matrix tags i.e. each term must have its associated tag. The caller will
            *not* process the output tags beyond the sentence length i.e. you can have
            arbitrary values beyond length.
        """
        # return numpy.zeros_like(terms)
        logits = self.sess.run(self.logits, {self.x: terms, self.lengths: lengths})
        return numpy.argmax(logits, axis=2)

    # TODO(student): You must implement this.
    def build_training(self):
        """Prepares the class for training.
        It is up to you how you implement this function, as long as train_on_batch
        works.
        Hint:
            - Lookup tf.contrib.seq2seq.sequence_loss
            - tf.losses.get_total_loss() should return a valid tensor (without raising
                an exception). Equivalently, tf.losses.get_losses() should return a
                non-empty list.
        """
        # pass
        weights = self.lengths_vector_to_binary_matrix(self.lengths)
        self.target = tf.placeholder(tf.int64, [None, self.max_length], 'target')
        loss = tf.contrib.seq2seq.sequence_loss(self.logits, self.target, weights)
        tf.losses.add_loss(loss)
        self.learning_rate = tf.placeholder_with_default(
            numpy.array(0.01, dtype='float32'), shape=[], name='learn_rate')
        opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = tf.contrib.training.create_train_op(tf.losses.get_total_loss(), opt)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def train_epoch(self, terms, tags, lengths, batch_size=32, learn_rate=1e-7):
        """Performs updates on the model given training training data.
        This will be called with numpy arrays similar to the ones created in
        Args:
            terms: int64 numpy array of size (# sentences, max sentence length)
            tags: int64 numpy array of size (# sentences, max sentence length)
            lengths:
            batch_size: int indicating batch size. Grader script will not pass this,
                but it is only here so that you can experiment with a "good batch size"
                from your main block.
            learn_rate: float for learning rate. Grader script will not pass this,
                but it is only here so that you can experiment with a "good learn rate"
                from your main block.
        Return:
            boolean. You should return True iff you want the training to continue. If
            you return False (or do not return anyhting) then training will stop after
            the first iteration!
        """
        # <-- Your implementation goes here.
        # Finally, make sure you uncomment the `return True` below.
        # return True
        # pass
        # self.sess = tf.Session()
        # self.sess.run(tf.global_variables_initializer())

        # self.sess.run(tf.initialize_all_variables())
        learn_rate = 0.01
        batch_size = 100
        indices = numpy.random.permutation(terms.shape[0])
        for si in range(0, terms.shape[0], batch_size):
            se = min(si + batch_size, terms.shape[0])

            slice_x = terms[indices[si:se]] + 0  # + 0 to copy slice

        self.sess.run(self.train_op, {
            self.x: slice_x,
            self.target: tags[indices[si:se]],
            self.learning_rate: learn_rate,
            self.lengths: lengths[indices[si:se]]
        })

        return True

    # TODO(student): You can implement this to help you, but we will not call it.
    def evaluate(self, terms, tags, lengths):
        pass


def main():
    """This will never be called by us, but you are encouraged to implement it for
    local debugging e.g. to get a good model and good hyper-parameters (learning
    rate, batch size, etc)."""
    # Read dataset.
    """
    reader = DatasetReader
    train_filename = sys.argv[1]
    test_filename = train_filename.replace('_train_', '_dev_')
    term_index, tag_index, train_data, test_data = reader.ReadData(train_filename, test_filename)
    (train_terms, train_tags, train_lengths) = train_data
    (test_terms, test_tags, test_lengths) = test_data
    model = SequenceModel(train_tags.shape[1], len(term_index), len(tag_index))
    model.build_inference()
    model.build_training()
    for j in range(10):
        model.train_epoch(train_terms, train_tags, train_lengths)
        print('Finished epoch %i. Evaluating ...' % (j + 1))
        model.evaluate(test_terms, test_tags, test_lengths)

    """

    MAX_TRAIN_TIME = 3 * 60  # 3 minutes.

    reader = DatasetReader
    # train_filename = 'hmm-training-data/ja_gsd_train_tagged.txt'
    train_filename = 'hmm-training-data/it_isdt_train_tagged.txt'
    test_filename = train_filename.replace('_train_', '_dev_')
    term_index, tag_index, train_data, test_data = reader.ReadData(train_filename, test_filename)
    (train_terms, train_tags, train_lengths) = train_data
    (test_terms, test_tags, test_lengths) = test_data

    num_terms = max(train_terms.max(), test_terms.max()) + 1
    model = SequenceModel(train_terms.shape[1], num_terms, train_tags.max() + 1)

    #
    def get_test_accuracy():
        predicted_tags = model.run_inference(test_terms, test_lengths)
        if predicted_tags is None:
            print('Is your run_inference function implented?')
            return 0
        test_accuracy = numpy.sum(
            numpy.cumsum(numpy.equal(test_tags, predicted_tags), axis=1)[
                numpy.arange(test_lengths.shape[0]), test_lengths]) / numpy.sum(test_lengths + 0.0)
        return test_accuracy

    model.build_inference()
    model.build_training()
    start_time_sec = time.clock()
    train_more = True
    num_iters = 0
    while train_more:
        print('  Test accuracy for after %i iterations is %f' % (num_iters, get_test_accuracy()))
        train_more = model.train_epoch(train_terms, train_tags, train_lengths)
        train_more = train_more and (time.clock() - start_time_sec) < MAX_TRAIN_TIME
        num_iters += 1

    # Done training. Measure test.
    print('  Final accuracy for  is %f' % (get_test_accuracy()))


if __name__ == '__main__':
    main()