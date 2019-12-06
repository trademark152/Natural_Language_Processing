import tensorflow as tf
print(tf.__version__)

# load sample dataset
mnist = tf.keras.datasets.mnist  # 28*28 images of hand-written digits 0-9

# load dataset to variables
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_train[0])

# scaling data:
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# plot data
import matplotlib.pyplot as plt
plt.imshow(x_train[0], cmap = plt.cm.binary)
# plt.show()

# feed forward network ARCHITECTURE
model = tf.keras.models.Sequential()

# add first layer: need to flatten the matrix into 1D vector
model.add(tf.keras.layers.Flatten())

# add hidden layers:
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))  # 128 neuron in the first hidden layers
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dropout(0.2))  # configure dropout

# add output layer:
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax)) # 10 digits

# Define parameters
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']) # 'adam' is the default go-to optimizer

# train the model
model.fit(x_train, y_train,epochs=3)

# calculate validation accuracy
val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss, val_acc)

# save model
model.save('epic_num_reader.model')

# load model:
new_model = tf.keras.models.load_model('epic_num_reader.model')

# do prediction: always input is a list
predictions = new_model.predict([x_test])

import numpy as np
print(np.argmax(predictions[0]))

# show results
plt.imshow(x_test[0])
plt.show()