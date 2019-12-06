import tensorflow as tf

"""
Tensor: Node in the DAG.
    Leaf nodes (i.e. without incoming edges) can be “Placeholders”, “Dataset Readers”, or “Variables”
    Intermediate nodes (with incoming edges) are Expressions of Tensors.
Gradients
    E.g. grads = tf.gradients(y, [a])[0]
Assign Operators.
    Applicable to Variables but not Placeholders
Collections
    While constructing Model, most functions automatically adds Tensors to collections. Model Optimization retrieves Tensors from collections.
    tf.global_variables()
    tf.trainable_variables()  # By default, all variables are made “trainable”
    tf.losses.get_regularization_losses()
    tf.losses.get_losses()
Summary Operators [Allows on TensorBoard, not in this talk, but on here].
Save and Restore Operators [Not on this talk, but here]
    We will use Pickle to save and restore. [Recall, we focus on barebones TensorFlow]
    
Make is_training to be a Boolean Tensor.
    During training steps: feed as True. During measuring accuracy on validation: feed as False.
    In the former, it will do BatchNorm and Dropout. In the latter, it uses BatchNorm statistics and no Dropout.
Layers.
    fully-connected layer takes as input matrix of activations ℝb x d0, multiplies by (trainable variable) kernel matrix ℝd0 x d1, adds (trainable variable) bias vector ℝd1, passes through Batchnorm, then through element-wise nonlinearity. Also, add L2 regularization penalty on the kernel matrix.
    The Kernel Matrix should not be initialized to all zero (like our toy example). Instead, 
    You can do each of the above as a separate line (using “pure barebones”), or can do all in one-line using tf.contrib.layers.fully_connected
Saving and Restoring Variables
I personally use Pickle!

"""
## EXAMPLE 1: simple operation
# type of tensor and data type
a = tf.placeholder(tf.float32, []) # leaf nodes
b = tf.placeholder(tf.float32, [])

# perform operation of nodes in DAG
y = a * b

# Starts C++ Backend
sess = tf.Session()

# run it
v = sess.run(y, {a: 2.5, b: 3.0})
print(v)


## EXAMPLE 2: toy optimization SUM(min(xi-w)^2)
# [None] means variable size
x_list = tf.placeholder(tf.float32, [None], name='x')
w = tf.Variable(0.0, name='w') # initialize weights

# calculate error
error = (w - x_list) ** 2   # error is a vector (not scalar).

# get the gradient of error wrt weight
grads = tf.gradients(error, [w])   # derivative of error w.r.t. scalar is a scalar. Sum implied
gw = grads[0]
learn_rate = 0.05

# assign new value of weight: applicable to variables but not placeholders
assign_op = w.assign(w - gw * learn_rate)

# start session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# loop through number of iterations
for i in range(100):
    sess.run(assign_op, {x_list: [1,2,3,4,5]})  # Mean should be 3.0

print('Converged to %f' % sess.run(w))


## EXAMPLE 3: MNIST example
# load KERAS dataset MNIST
mnist = tf.keras.datasets.mnist

# load data into variables
(x_train, y_train),(x_test, y_test) = mnist.load_data()

# normalize data
x_train, x_test = x_train / 255.0, x_test / 255.0

# Build model
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

# Specify model parameters
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# train model
model.fit(x_train, y_train, epochs=5)

# test model
model.evaluate(x_test, y_test)
