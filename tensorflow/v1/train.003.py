#  Example 3: Recurrent Neural Network (RNN) for Text Classification


import tensorflow as tf
from tensorflow.keras import layers

# Load and preprocess the IMDB movie review dataset
imdb = tf.keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
train_data = tf.keras.preprocessing.sequence.pad_sequences(train_data, maxlen=250)
test_data = tf.keras.preprocessing.sequence.pad_sequences(test_data, maxlen=250)

# Define the RNN architecture
model = tf.keras.Sequential()
model.add(layers.Embedding(10000, 16))
model.add(layers.SimpleRNN(32))
model.add(layers.Dense(1, activation='sigmoid'))

# Compile and train the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit(train_data, train_labels, epochs=10, validation_data=(test_data, test_labels))

