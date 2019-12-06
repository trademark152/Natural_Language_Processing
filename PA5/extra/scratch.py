import numpy as np
import tensorflow as tf
# a = [1,2]
# a.extend([0]*(10-len(a)))
# print(a)

lengths = [4, 3, 5, 2]
lengths_transposed = tf.expand_dims(lengths, 1)
# Make a 4 x 8 matrix where each row contains [0, 1, ..., 7]
range = tf.range(0, 8, 1)
range_row = tf.expand_dims(range, 0)

# Use the logical operations to create a mask
mask = tf.less(range_row, lengths_transposed)

# Use the select operation to select between 1 or 0 for each value.
result = tf.where(mask, tf.ones([4, 8]), tf.zeros([4, 8]))

with tf.Session() as sess:
    print("Length transposed: ", sess.run(lengths_transposed))
    print("range: ", sess.run(range))
    print("range_row: ", sess.run(range_row))
    print("mask: ", sess.run(mask))
    print("result: ", sess.run(result))