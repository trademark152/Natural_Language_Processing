import time
import os
import sys

import numpy

MAX_TRAIN_TIME = 3 * 60  # 3 minutes.

num_terms = max(train_terms.max(), test_terms.max()) + 1
model = starter.SequenceModel(train_terms.shape[1], num_terms, train_tags.max() + 1)


# function to get test accuracy
def get_test_accuracy():
    predicted_tags = model.run_inference(test_terms, test_lengths)
    if predicted_tags is None:
        print('Is your run_inference function implented?')
        return 0
    test_accuracy = numpy.sum(
        numpy.cumsum(test_tags == predicted_tags, axis=1)[
            numpy.arange(test_lengths.shape[0]), test_lengths]) / numpy.sum(test_lengths + 0.0)
    return test_accuracy


model.build_inference()
model.build_training()
start_time_sec = time.clock()
train_more = True
num_iters = 0
while train_more:
    print('  Test accuracy for %s after %i iterations is %f' % (language_name, num_iters, get_test_accuracy()))
    train_more = model.train_epoch(train_terms, train_tags, train_lengths)
    train_more = train_more and (time.clock() - start_time_sec) < MAX_TRAIN_TIME
    num_iters += 1

# Done training. Measure test.
print('  Final accuracy for %s is %f' % (language_name, get_test_accuracy()))
# <span style="text-decoration:underline"><strong> Note: By the end of the weekend, the last accuracy will be evaluated in a NEW program</strong></span>
# <span style="text-decoration:underline"><strong> (i.e. calling save_model in one program and load_model then measure test accuracy in another program)</strong></span>