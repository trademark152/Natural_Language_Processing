"""
Tensorflow program consists of constants, variables and placeholders; representing different parameters of deep learning model

Constant: fixed value tensors - not trainable, defined as tf.constant(2)
Variables: tensors initialized in a session: trainable, defined as tf.variable()
Placeholders: tensors whose values are unknown during the graph construction but passed as input during a session. This way you can feed data from outside world to the computational graph

"""

""" Sample code for constants"""
import numpy as np
import tensorflow as tf

a = tf.constant(3)
b = tf.constant(5)
c = a+b

# to print c you have to initialize graph construction in an interactive session (useful in jupyter notebook)
tf.InteractiveSession()
print(c.eval())

# another way to print c
with tf.Session() as sess:
    print(sess.run(c))


""" Sample code for variable"""
a1 = tf.Variable(0)
b1 = tf.constant(1)
mid = tf.add(a1, b1)
final = tf.assign(a1, mid) # assign "mid" to a1

# IMPORTANT: need to initialize all variables before the session
var = tf.initialize_all_variables()

# start a session
with tf.Session() as sess:
    # run the initalization first
    sess.run(var)

    for i in range(3):
        # perform the computational graph in tf by tracing backwards operation

        # for example, when i=0
        # trace back to "final": assign "mid" to "a1". To perform this, trace back to "mid" and "a1"
        # In "mid", perform addition between "a1" and "b1". To perform this, trace back to "a1" and "b1"
        # In "a1", a1 = 0; In "b1", b1 = 1
        sess.run(final)
        print("a1: ", sess.run(a1))


""" Sample code for placeholder"""
x = tf.placeholder("float", [None, 3])  # a 2D array with any row and 3 columns
y = x**2

# run a session
with tf.Session() as sess:
    x_data = [[1,2,3],[4,5,6]]

    # this command feeds x_data to placeholder x to calculate y
    # feed_dict is a python dictionary mapping from tf.placeholder variables to data(np array, list)
    result = sess.run(y, feed_dict={x: x_data})
    print(result)