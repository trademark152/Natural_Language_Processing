import tensorflow as tf

## EXAMPLE 1
# type of tensor and data type
a = tf.placeholder(tf.float32, [])
b = tf.placeholder(tf.float32, [])
y = a * b

sess = tf.Session()  # Starts C++ Backend
sess.run(y, {a:2.5, b:3.0})


## EXAMPLE 2
# [None] means variable size
x_list = tf.placeholder(tf.float32, [None], name='x')
w = tf.Variable(0.0, name='w')

error = (w - x_list) ** 2   # Note broadcasting. error is a vector (not scalar).
grads = tf.gradients(error, [w])   # derivative of error w.r.t. scalar is a scalar. Sum implied
gw = grads[0]
learn_rate = 0.05

assign_op = w.assign(w - gw * learn_rate)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in xrange(100):
  sess.run(assign_op, {x_list: [1,2,3,4,5]})  # Mean should be 3.0

print ('Converged to %f' % sess.run(w))


## EXAMPLE 3
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)
