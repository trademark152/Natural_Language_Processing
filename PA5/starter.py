"""
USC_EMAIL: tranmt@usc.edu
PASSWORD: a9e0bce1c7c918fe
"""


USC_EMAIL = 'tranmt@usc.edu'  # TODO(student): Fill to compete on rankings.
PASSWORD = 'a9e0bce1c7c918fe'  # TODO(student): You will be given a password via email.

# curl http://sami.haija.org/expandspaces.py | FILE=starter.py  python??

import numpy as np
import tensorflow as tf
import sys
import numpy as np
import pickle

from __future__ import print_function, division
import matplotlib.pyplot as plt

num_epochs = 100
total_series_length = 50000
truncated_backprop_length = 15
state_size = 4
num_classes = 2
echo_step = 3
batch_size = 5
num_batches = total_series_length//batch_size//truncated_backprop_length

"""" DECLARE VARIABLES"""
beginTag = 'q0' # beginning of sentences tag
endTag = 'qE'  # end of sentences tag
currentTag = ''  # current tag
line = ''  # current line
parsedSentences = []

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

        Return:
          The parsed file as a list of lists: [parsedLine1, parsedLine2, ...]
          each parsedLine is a list: [(term1, tag1), (term2, tag2), ...]
        """
        # pass
        # read file
        wordTagList = []  # dict of {word|tag pair: occurrences}
        text = open(filename, 'r', encoding="utf8")
        for line in text:
            wordTagList.append(line.split())

        # print("wordTagList: ", wordTagList[:100])
        text.close()

        # initialize
        parsedFile = []
        idxTag = len(tag_index.keys())
        idxWord = len(term_index.keys())

        for line in wordTagList:
            parsedLine = []
            parsedSentence = []

            # Tag counter by looping through each word|tag in
            for wordTag in line:
                # print(wordTag)
                # find index to separate word from tag
                idx = 0
                # loop through each position of word|tag
                for pos in range(len(wordTag) - 1, 0, -1):
                    # stop when the slash is met
                    if wordTag[pos] == '/':
                        idx = pos
                        break

                # Use the index to separate the tag from '/' to end
                currentTag = wordTag[idx + 1:]
                currentWord = wordTag[:idx]

                # update tag dictionary
                if currentTag not in tag_index.keys():
                    tag_index[currentTag] = idxTag

                    # update tag index
                    idxTag += 1

                # update word dictionary
                if currentWord not in term_index.keys():
                    term_index[currentWord] = idxWord

                    # update word index
                    idxWord += 1

                # update parsedLine with indexes
                parsedLine.append((term_index[currentWord], tag_index[currentTag]))
                # parsedSentence.append((currentWord,currentTag))

            # update parseFile
            parsedFile.append(parsedLine)
            # parsedSentences.append(parsedSentence)

        # print("parsedFile: ", parsedFile[:10])
        # print("term index: ", term_index)
        # print("tag index: ", tag_index)

        return parsedFile

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
        # pass
        # print(dataset[:5])
        # number of sentences
        N = len(dataset)
        # print("number of sentences: ", N)

        # maximum sentence length
        # print(max(dataset, key=len))
        # print(max(parsedSentences, key = len)) # print longest sentence
        T = len(max(dataset, key=len))
        # print("length of longest sentence: ", T)

        terms_matrix = np.empty([N, T], dtype=np.int64)
        tags_matrix = np.empty([N, T], dtype=np.int64)
        lengths_arr = np.empty([N], dtype=np.int64)

        # loop through each line
        for idx, line in enumerate(dataset):
            # update the length of each line
            lengths_arr[idx] = len(line)

            # update the terms_matrix with the terms of each line
            terms_line = [i[0] for i in line]

            # pad 0s at the end
            terms_line.extend([0] * (T - len(line)))
            # print("terms_line: ", terms_line)
            # print(T-len(line))
            terms_matrix[idx] = terms_line

            # update the tags_matrix with the tags of each line
            tags_line = [i[1] for i in line]
            tags_line.extend([0] * (T - len(line)))
            tags_matrix[idx] = tags_line


        # print("terms_matrix: ", terms_matrix)
        # print("tags_matrix: ", tags_matrix)
        # print("lengths_arr: ", lengths_arr)
        return (terms_matrix, tags_matrix, lengths_arr)

    @staticmethod
    def ReadData(train_filename, test_filename=None):
        """Returns numpy arrays and indices for train (and optionally test) data.

          NOTE: Please do not change this method. The grader will use an identitical copy of this method (if you change this, your offline testing will no longer match the grader).

          Args:
            train_filename: .txt path containing training data, one line per sentence.
              The data must be tagged (i.e. "word1/tag1 word2/tag2 ...").
            test_filename: Optional .txt path containing test data.

          Returns:
            A tuple of 3-elements or 4-elements, the later iff test_filename is given.
            The first 2 elements are term_index and tag_index, which are dictionaries, respectively, from term to integer ID and from tag to integer ID. The int
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
        sess = tf.InteractiveSession()  # initializes a tensorflow session

        self.max_length = max_length
        self.num_terms = num_terms
        self.num_tags = num_tags
        self.x = tf.placeholder(tf.int64, [None, self.max_length], 'X')
        self.lengths = tf.placeholder(tf.int32, [None], 'lengths')


    # TODO(student): You must implement this.
    def lengths_vector_to_binary_matrix(self, length_vector):
        """Returns a binary mask (as float32 tensor) from (vector) int64 tensor.

        Specifically, the return matrix B will have the following:
          B[i, :lengths[i]] = 1 and B[i, lengths[i]:] = 0 for each i.
        However, since we are using tensorflow rather than numpy in this function,
        you cannot set the range as described.
        """

        return tf.ones([tf.shape(length_vector), self.max_length], dtype=tf.float32)

    # TODO(student): You must implement this.
    def save_model(self, filename):
        """Saves model to a file."""
        # pass
        var_dict = {v.name: v for v in tf.global_variables()}
        sess = tf.Session()
        pickle.dump(sess.run(var_dict), open(filename, 'w'))

    # TODO(student): You must implement this.
    def load_model(self, filename):
        """Loads model from a file."""
        # pass
        var_values = pickle.load(open(filename))
        assign_ops = [v.assign(var_values[v.name]) for v in tf.global_variables()]
        sess = tf.Session()
        sess.run(assign_ops)

    # TODO(student): You must implement this.

    def build_inference(self):
        """Build the expression from (self.x, self.lengths) to (self.logits).

        Please do not change or override self.x nor self.lengths in this function.

        Hint:
          - Use lengths_vector_to_binary_matrix
          - You might use tf.reshape, tf.cast, and/or tensor broadcasting.
        """
        # TODO(student): make logits an RNN on x.
        pass

    # TODO(student): You must implement this.
    # def run_inference(self, tags, lengths):
    def run_inference(self, terms, lengths):
        """Evaluates self.logits given self.x and self.lengths. Get the most probable tags for each word

        Hint: This function is straight forward and you might find this code useful:
        # logits = session.run(self.logits, {self.x: tags, self.lengths: lengths})
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
        session = tf.Session()
        logits = session.run(self.logits, {self.x: terms, self.lengths: lengths})
        return numpy.argmax(logits, axis=2)
        # return numpy.zeros_like(terms)


    # TODO(student): You must implement this.
    def build_training(self):
        """Prepares the class for training.

        It is up to you how you implement this function, as long as train_epoch works.

        Hint:
          - Lookup tf.contrib.seq2seq.sequence_loss
          - tf.losses.get_total_loss() should return a valid tensor (without raising an exception). Equivalently, tf.losses.get_losses() should return a non-empty list.
        """
        # pass

        # # Compute loss
        # sigmoid = tf.nn.sigmoid(self.logits)
        # cross_entropy = -self.y_onehot * tf.log(sigmoid + 1e-6) - (1 - self.y_onehot) * tf.log(1 - sigmoid + 1e-6)
        # cross_entropy = tf.reduce_sum(cross_entropy, axis=2)  # Remove the label axes
        #
        # # Mask it by lengths.
        # csum_cross_entropy = tf.math.cumsum(cross_entropy, axis=1)
        # loss = tf.reduce_sum(csum_cross_entropy * self.lengths_onehot)

        self.loss = tf.contrib.seq2seq.sequence_loss(
            self.logits,
            self.targets,
            self.binaryMask,
            average_across_timesteps=True,
            average_across_batch=True,
            softmax_loss_function=None,
            name=None
        )

        print("Total loss: ", tf.losses.get_total_loss())
        print("loss: ", self.loss)

        self.train_step = tf.train.AdagradOptimizer(0.3).minimize(self.loss)

        return self.loss



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
        """
        # pass
        init_state = tf.placeholder(tf.float32, [batch_size, self.state_size])
        _current_state = np.zeros((batch_size, self.state_size))

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            loss_list = []

            _total_loss, _train_step, _current_state, _predictions_series = sess.run(
                [self.loss, self.train_step, self.states[-1], self.predictions_series],
                feed_dict={
                    terms,
                    tags})


    # TODO(student): You can implement this to help you, but we will not call it.
    def evaluate(self, terms, tags, lengths):
        pass



"""
# Finally, your model will be tested by the grader script, which invokes 2 programs, the training
program then the testing program. They will run as:
# TRAINING PROGRAM: 
model = SequenceModel (max length , num terms , num tags )
model . b u i l d i n f e r e n c e ( )
model . b u i l d t r a i n i n g ( )
while ( t ime spent < K) :
model . t r a in epo ch ( terms , tags , l eng ths )
model . save model ( ' /some/ f i l e /path ' )
# TESTING PROGRAM [runs in a separate shell command , after training program]
model = SequenceModel (max length , num terms , num tags )
model . load model ( ' /some/ f i l e /path ' )
model . buildinference()
model . runinference(testtags,testlength) # and compare with ground -truth
"""
def main():
    """This will never be called by us, but you are encouraged to implement it for
    local debugging e.g. to get a good model and good hyper-parameters (learning
    rate, batch size, etc)."""
    # Read dataset.
    reader = DatasetReader()

    # get train file name from command line
    train_filename = sys.argv[1]

    # get test file
    test_filename = train_filename.replace('_train_', '_dev_')

    # read data file
    term_index, tag_index, train_data, test_data = reader.ReadData(train_filename, test_filename)
    (train_terms, train_tags, train_lengths) = train_data
    # print("train_data: ", train_data)
    (test_terms, test_tags, test_lengths) = test_data

    # Create sequence models:
    print("train_tags.shape: ", train_tags.shape)
    model = SequenceModel(train_tags.shape[1], len(term_index), len(tag_index))

    model.build_inference()
    model.build_training()

    # train 10 epochs
    for j in tf.range(10):
        model.train_epoch(train_terms, train_tags, train_lengths)
        print('Finished epoch %i. Evaluating ...' % (j + 1))

        # evaluate model
        model.evaluate(test_terms, test_tags, test_lengths)


if __name__ == '__main__':
    main()

"""
NOTE
ReadFile must be returning the dataset as integer IDs (not the string terms and tags!): 

each parsedLine is a list: [(term<strong>Id</strong>1, tag<strong>Id</strong>1), (termId2, tagId2), ...]. In addition, I made a comment that the _index variables can be partially pre-populated, and you are only expected to store new tags/terms onto the indices with a new integer and leaving no int gaps.

run_inference had a typo in the parameter name. We are passing terms and not the tags [the old documentation said that it should be terms but the variable name was wrong].
Your implementation of train_epoch must return True if you want it to be called again. The timing budget is 3 minutes for training. If it returns False, then it wont be called again (i.e. you would train a single epoch, which in my trials, gets me almost all the way to training 10 or 20 epochs, if you use a good optimizer). It defaults to not returning anything (i.e. identical to returning False) to make the grading script act very fast for those who are still at task 1 [i.e. they would get their grade in seconds rather than >9 minutes == 3 minutes per language].

"""