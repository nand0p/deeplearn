#Example 1: Linear Regression

import tensorflow as tf
import numpy as np

# Create some random data
x = np.random.rand(100).astype(np.float32)
y = x * 0.1 + 0.3

# Create TensorFlow variables for weight and bias
w = tf.Variable(tf.random.uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))

# Define the linear regression model
y_pred = w * x + b

# Define the loss function (mean squared error)
loss = tf.reduce_mean(tf.square(y_pred - y))

# Create the optimizer and minimize the loss
optimizer = tf.optimizers.SGD(learning_rate=0.5)
train = optimizer.minimize(loss)

# Initialize the variables
init = tf.compat.v1.global_variables_initializer()

# Start a TensorFlow session and run the training
with tf.compat.v1.Session() as sess:
    sess.run(init)
    for step in range(201):
        sess.run(train)
        if step % 20 == 0:
            print(step, sess.run(w), sess.run(b), sess.run(loss))
